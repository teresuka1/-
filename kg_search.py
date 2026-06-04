from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENTITY_PATH = BASE_DIR / "data" / "ds_entities_disambiguated.json"
DEFAULT_RELATION_PATH = BASE_DIR / "data" / "ds_relations.json"


INTENT_PREDICATES = {
    "包含": ("哪些", "什么", "包括", "包含", "分为", "类型", "种类", "组成"),
    "属于": ("属于", "是什么", "哪类", "类型", "种类"),
    "支持操作": ("支持", "操作", "插入", "删除", "查找", "修改", "入栈", "出栈", "入队", "出队"),
    "采用存储结构": ("采用", "使用", "存储", "顺序存储", "链式存储"),
    "具有性质": ("性质", "特点", "规则", "特征", "先进先出", "先进后出"),
    "应用于": ("用于", "应用", "场景", "解决", "问题"),
    "求解问题": ("用于", "应用", "解决", "求", "算法", "问题"),
    "组成": ("组成", "构成", "由", "包括"),
    "相关": ("相关", "关系", "区别", "联系"),
}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    category: str
    context: str
    mention_count: int


@dataclass(frozen=True)
class SearchMatch:
    subject: str
    predicate: str
    object: str
    subject_category: str
    object_category: str
    confidence: float
    evidences: list[dict[str, Any]]
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_category": self.subject_category,
            "object_category": self.object_category,
            "confidence": self.confidence,
            "evidences": self.evidences,
            "score": round(self.score, 4),
            "source": self.source,
        }


class KnowledgeGraphSearch:
    def __init__(
        self,
        entity_path: Path | str = DEFAULT_ENTITY_PATH,
        relation_path: Path | str = DEFAULT_RELATION_PATH,
    ) -> None:
        self.entity_path = Path(entity_path)
        self.relation_path = Path(relation_path)
        self.entities = self._load_entities(self.entity_path)
        self.relations = self._load_relations(self.relation_path)
        self.entity_by_id = {entity.entity_id: entity for entity in self.entities}
        self.entity_names = sorted(
            {entity.name for entity in self.entities if entity.name},
            key=len,
            reverse=True,
        )
        self.relations_by_entity_id: dict[str, list[int]] = defaultdict(list)
        for index, relation in enumerate(self.relations):
            for entity_id_key in ("subject_id", "object_id"):
                entity_id = str(relation.get(entity_id_key, ""))
                if entity_id:
                    self.relations_by_entity_id[entity_id].append(index)
            relation["_search_text"] = self._relation_search_text(relation)
            relation["_search_vector"] = ngram_vector(relation["_search_text"])

    def search(self, question: str, limit: int = 8) -> list[SearchMatch]:
        question = normalize_text(question)
        if not question:
            return []

        query_vector = ngram_vector(question)
        matched_entities = self._match_entities(question)
        direct_scores = self._score_relations(question, query_vector, matched_entities)
        expanded_scores = self._expand_one_hop(matched_entities, direct_scores)

        merged: dict[int, tuple[float, str]] = {}
        for index, score in direct_scores.items():
            merged[index] = (score, "direct")
        for index, score in expanded_scores.items():
            previous = merged.get(index)
            if previous is None or score > previous[0]:
                merged[index] = (score, "expanded")

        ranked = sorted(
            merged.items(),
            key=lambda item: (
                item[1][0],
                float(self.relations[item[0]].get("confidence", 0)),
                int(self.relations[item[0]].get("evidence_count", 0)),
            ),
            reverse=True,
        )

        return [
            self._to_match(index, score=score, source=source)
            for index, (score, source) in ranked[:limit]
            if score > 0
        ]

    def has_direct_evidence(self, matches: list[SearchMatch], threshold: float = 2.2) -> bool:
        return any(match.source == "direct" and match.score >= threshold for match in matches)

    def build_context(self, question: str, matches: list[SearchMatch], limit: int = 8) -> str:
        if not matches:
            return "未检索到匹配的知识图谱关系。"

        lines = []
        for index, match in enumerate(matches[:limit], start=1):
            evidence_texts = [
                normalize_text(evidence.get("text", ""))
                for evidence in match.evidences
                if normalize_text(evidence.get("text", ""))
            ]
            evidence_preview = " / ".join(dict.fromkeys(evidence_texts[:2])) or "无证据句"
            lines.append(
                f"{index}. ({match.subject_category}){match.subject} "
                f"-[{match.predicate}]-> ({match.object_category}){match.object}；"
                f"置信度 {match.confidence:.2f}；检索分 {match.score:.2f}；"
                f"证据：{evidence_preview}"
            )
        return "\n".join(lines)

    def _load_entities(self, path: Path) -> list[Entity]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_entities = data.get("entities", []) if isinstance(data, dict) else data
        entities = []
        for raw in raw_entities:
            name = normalize_text(raw.get("canonical_name") or raw.get("text"))
            category = normalize_text(raw.get("disambiguated_category") or raw.get("category"))
            entity_id = normalize_text(raw.get("linked_entity_id") or raw.get("entity_id"))
            if not entity_id and category and name:
                entity_id = f"{category}:{name}"
            entities.append(
                Entity(
                    entity_id=entity_id,
                    name=name,
                    category=category,
                    context=normalize_text(raw.get("context")),
                    mention_count=int(raw.get("mention_count") or 0),
                )
            )
        return entities

    def _load_relations(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        relations = data.get("relations", []) if isinstance(data, dict) else data
        return [dict(relation) for relation in relations]

    def _match_entities(self, question: str) -> dict[str, float]:
        normalized_question = normalize_key(question)
        matches: dict[str, float] = {}
        for entity in self.entities:
            name_key = normalize_key(entity.name)
            if not name_key:
                continue
            if name_key in normalized_question:
                length_bonus = min(len(name_key) / 8, 1.0)
                popularity_bonus = min(math.log(entity.mention_count + 1) / 8, 0.4)
                matches[entity.entity_id] = max(
                    matches.get(entity.entity_id, 0),
                    3.0 + length_bonus + popularity_bonus,
                )
            elif normalized_question and normalized_question in name_key:
                matches[entity.entity_id] = max(matches.get(entity.entity_id, 0), 1.4)
        return matches

    def _score_relations(
        self,
        question: str,
        query_vector: Counter[str],
        matched_entities: dict[str, float],
    ) -> dict[int, float]:
        scores = {}
        question_key = normalize_key(question)
        for index, relation in enumerate(self.relations):
            score = 0.0
            subject_id = normalize_text(relation.get("subject_id"))
            object_id = normalize_text(relation.get("object_id"))
            subject = normalize_text(relation.get("subject"))
            object_ = normalize_text(relation.get("object"))
            predicate = normalize_text(relation.get("predicate"))

            if subject_id in matched_entities:
                score += matched_entities[subject_id]
            if object_id in matched_entities:
                score += matched_entities[object_id]
            if normalize_key(subject) in question_key and subject:
                score += 1.5
            if normalize_key(object_) in question_key and object_:
                score += 1.5
            if predicate and predicate in question:
                score += 1.0
            score += self._predicate_intent_boost(question, predicate)

            relation_similarity = cosine_similarity(query_vector, relation["_search_vector"])
            if relation_similarity:
                score += relation_similarity * 3.0

            evidence_similarity = max(
                (
                    cosine_similarity(query_vector, ngram_vector(evidence.get("text", "")))
                    for evidence in relation.get("evidences", [])
                ),
                default=0.0,
            )
            score += evidence_similarity * 2.0

            confidence = float(relation.get("confidence") or 0)
            score += confidence * 0.25

            if score >= 1.0:
                scores[index] = score
        return scores

    def _expand_one_hop(
        self,
        matched_entities: dict[str, float],
        direct_scores: dict[int, float],
    ) -> dict[int, float]:
        seed_entity_ids = set(matched_entities)
        for index, score in sorted(direct_scores.items(), key=lambda item: item[1], reverse=True)[:4]:
            if score < 2.0:
                continue
            relation = self.relations[index]
            seed_entity_ids.add(normalize_text(relation.get("subject_id")))
            seed_entity_ids.add(normalize_text(relation.get("object_id")))

        expanded_scores: dict[int, float] = {}
        for entity_id in seed_entity_ids:
            if not entity_id:
                continue
            for index in self.relations_by_entity_id.get(entity_id, []):
                relation = self.relations[index]
                confidence = float(relation.get("confidence") or 0)
                evidence_count = int(relation.get("evidence_count") or 0)
                expanded_scores[index] = max(
                    expanded_scores.get(index, 0.0),
                    0.9 + confidence * 0.35 + min(evidence_count, 3) * 0.1,
                )
        return expanded_scores

    def _predicate_intent_boost(self, question: str, predicate: str) -> float:
        keywords = INTENT_PREDICATES.get(predicate, ())
        return sum(0.35 for keyword in keywords if keyword and keyword in question)

    def _relation_search_text(self, relation: dict[str, Any]) -> str:
        evidence_text = " ".join(
            normalize_text(evidence.get("text", ""))
            for evidence in relation.get("evidences", [])
            if isinstance(evidence, dict)
        )
        triggers = " ".join(str(trigger) for trigger in relation.get("triggers", []))
        return " ".join(
            [
                normalize_text(relation.get("subject")),
                normalize_text(relation.get("subject_category")),
                normalize_text(relation.get("predicate")),
                normalize_text(relation.get("object")),
                normalize_text(relation.get("object_category")),
                triggers,
                evidence_text,
            ]
        )

    def _to_match(self, index: int, score: float, source: str) -> SearchMatch:
        relation = self.relations[index]
        return SearchMatch(
            subject=normalize_text(relation.get("subject")),
            predicate=normalize_text(relation.get("predicate")),
            object=normalize_text(relation.get("object")),
            subject_category=normalize_text(relation.get("subject_category")),
            object_category=normalize_text(relation.get("object_category")),
            confidence=float(relation.get("confidence") or 0),
            evidences=[
                {
                    "text": normalize_text(evidence.get("text")),
                    "start": evidence.get("start"),
                    "end": evidence.get("end"),
                }
                for evidence in relation.get("evidences", [])
                if isinstance(evidence, dict)
            ],
            score=score,
            source=source,
        )


def normalize_text(text: Any) -> str:
    return str(text or "").replace("\ufeff", "").strip()


def normalize_key(text: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(text)).lower()


def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text).lower()
    return re.findall(r"[a-z0-9+]+|[\u4e00-\u9fff]", normalized)


def ngrams(tokens: list[str], min_n: int = 1, max_n: int = 3) -> list[str]:
    grams = []
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        grams.extend("".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return grams or tokens


def ngram_vector(text: Any) -> Counter[str]:
    return Counter(ngrams(tokenize(text)))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in common_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
