import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT = "data/ds_entities.json"
DEFAULT_DICT = "data/ds_domain_dict.json"
DEFAULT_OUTPUT = "data/ds_entities_disambiguated.json"
DEFAULT_CSV_OUTPUT = "data/ds_entities_disambiguated.csv"


@dataclass
class CandidateEntity:
    """A lightweight KB node built from the domain dictionary and mention contexts."""

    entity_id: str
    canonical_name: str
    category: str
    aliases: List[str] = field(default_factory=list)
    profile_terms: Counter = field(default_factory=Counter)
    popularity: int = 1


class TfidfVectorizer:
    """Small TF-IDF vectorizer for character n-grams, avoiding external packages."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)) -> None:
        self.ngram_range = ngram_range
        self.idf: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        tokens = re.findall(r"[A-Za-z0-9+]+|[\u4e00-\u9fff]|[^\s]", text)
        grams: List[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(tokens) < n:
                continue
            grams.extend("".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return grams or tokens

    def fit(self, documents: Sequence[str]) -> None:
        doc_count = max(len(documents), 1)
        df: Counter = Counter()
        for doc in documents:
            df.update(set(self.tokenize(doc)))
        self.idf = {
            term: math.log((doc_count + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

    def transform(self, text: str) -> Dict[str, float]:
        tf = Counter(self.tokenize(text))
        if not tf:
            return {}
        total = sum(tf.values())
        vector = {
            term: (count / total) * self.idf.get(term, 1.0)
            for term, count in tf.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return vector
        return {term: value / norm for term, value in vector.items()}


def normalize_text(text: object) -> str:
    return str(text or "").replace("\ufeff", "").strip()


def normalize_key(text: object) -> str:
    return re.sub(r"\s+", "", normalize_text(text)).lower()


def safe_json_load(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def load_extraction_result(path: Path) -> Tuple[dict, List[dict]]:
    data = safe_json_load(path, {"entities": []})
    if isinstance(data, list):
        return {"source_file": str(path), "entities": data}, data
    if isinstance(data, dict):
        entities = data.get("entities", [])
        return data, entities if isinstance(entities, list) else []
    return {"source_file": str(path), "entities": []}, []


def load_domain_dict(path: Path, entities: Sequence[dict]) -> Dict[str, List[str]]:
    data = safe_json_load(path, {})
    domain_dict: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for category, terms in data.items():
            if isinstance(terms, list):
                domain_dict[normalize_text(category)] = [
                    normalize_text(term) for term in terms if normalize_text(term)
                ]

    for entity in entities:
        category = normalize_text(entity.get("category"))
        text = normalize_text(entity.get("text"))
        if not category or not text:
            continue
        bucket = domain_dict.setdefault(category, [])
        if text not in bucket:
            bucket.append(text)
    return domain_dict


def span_value(entity: dict, field: str) -> int:
    try:
        return int(entity.get(field, -1))
    except (TypeError, ValueError):
        return -1


def is_nested(shorter: dict, longer: dict) -> bool:
    short_start = span_value(shorter, "start")
    short_end = span_value(shorter, "end")
    long_start = span_value(longer, "start")
    long_end = span_value(longer, "end")
    if min(short_start, short_end, long_start, long_end) < 0:
        return False
    if short_start == long_start and short_end == long_end:
        return False
    short_text = normalize_text(shorter.get("text"))
    long_text = normalize_text(longer.get("text"))
    if len(short_text) >= len(long_text):
        return False
    return long_start <= short_start and short_end <= long_end


def remove_nested_entities(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    sorted_entities = sorted(
        entities,
        key=lambda item: (
            span_value(item, "start"),
            -max(span_value(item, "end") - span_value(item, "start"), 0),
            -len(normalize_text(item.get("text"))),
        ),
    )
    kept: List[dict] = []
    removed = 0
    for entity in sorted_entities:
        if any(is_nested(entity, other) for other in sorted_entities):
            removed += 1
            continue
        kept.append(dict(entity))
    kept.sort(key=lambda item: (span_value(item, "start"), span_value(item, "end")))
    return kept, removed


def merge_duplicate_entities(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for entity in entities:
        text = normalize_text(entity.get("text"))
        if not text:
            continue
        groups[normalize_key(text)].append(dict(entity))

    merged: List[dict] = []
    duplicate_removed = 0
    for group in groups.values():
        group.sort(key=lambda item: (span_value(item, "start"), span_value(item, "end")))
        representative = dict(group[0])
        duplicate_removed += max(len(group) - 1, 0)

        category_counter = Counter(normalize_text(item.get("category")) for item in group)
        category_counter.pop("", None)
        if category_counter:
            representative["category"] = category_counter.most_common(1)[0][0]

        contexts = []
        for item in group:
            context = normalize_text(item.get("context"))
            if context and context not in contexts:
                contexts.append(context)
        if contexts:
            representative["context"] = " ".join(contexts[:5])

        representative["mention_count"] = len(group)
        representative["occurrences"] = [
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "category": item.get("category", ""),
                "context": item.get("context", ""),
            }
            for item in group
        ]
        merged.append(representative)

    merged.sort(key=lambda item: (span_value(item, "start"), span_value(item, "end")))
    return merged, duplicate_removed


def preprocess_entities(entities: Sequence[dict]) -> Tuple[List[dict], dict]:
    without_nested, nested_removed = remove_nested_entities(entities)
    merged, duplicate_removed = merge_duplicate_entities(without_nested)
    return merged, {
        "raw_entity_count": len(entities),
        "nested_entity_removed_count": nested_removed,
        "duplicate_entity_removed_count": duplicate_removed,
        "unique_entity_count": len(merged),
    }


def counter_to_text(counter: Counter) -> str:
    pieces: List[str] = []
    for term, count in counter.items():
        pieces.extend([term] * min(count, 8))
    return " ".join(pieces)


def build_knowledge_base(
    domain_dict: Dict[str, List[str]], entities: Sequence[dict]
) -> Dict[str, CandidateEntity]:
    candidates: Dict[str, CandidateEntity] = {}
    contexts_by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    freq_by_pair: Counter = Counter()

    for mention in entities:
        text = normalize_text(mention.get("text"))
        category = normalize_text(mention.get("category"))
        context = normalize_text(mention.get("context"))
        if not text or not category:
            continue
        contexts_by_pair[(category, text)].append(context)
        freq_by_pair[(category, text)] += int(mention.get("mention_count", 1) or 1)

    for category, terms in domain_dict.items():
        category_terms = [term for term in dict.fromkeys(terms) if term]
        for term in category_terms:
            entity_id = make_entity_id(category, term)
            aliases = build_aliases(term)
            profile = Counter()
            profile.update([term, category])
            profile.update(category_terms)
            for context in contexts_by_pair.get((category, term), []):
                profile.update(TfidfVectorizer().tokenize(context))
            candidates[entity_id] = CandidateEntity(
                entity_id=entity_id,
                canonical_name=term,
                category=category,
                aliases=aliases,
                profile_terms=profile,
                popularity=max(freq_by_pair.get((category, term), 0), 1),
            )
    return candidates


def make_entity_id(category: str, name: str) -> str:
    raw = f"{category}:{name}"
    return re.sub(r"\s+", "_", raw)


def build_aliases(name: str) -> List[str]:
    aliases = [name]
    ascii_parts = re.findall(r"[A-Za-z0-9+]+", name)
    if ascii_parts:
        aliases.append("".join(part[0] for part in ascii_parts if part))
        aliases.extend(ascii_parts)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def surface_match_score(mention_text: str, candidate: CandidateEntity) -> float:
    mention_key = normalize_key(mention_text)
    alias_keys = [normalize_key(alias) for alias in candidate.aliases]
    if mention_key in alias_keys:
        return 1.0
    if any(mention_key and (mention_key in alias or alias in mention_key) for alias in alias_keys):
        return 0.55
    mention_upper = re.sub(r"[^A-Z0-9+]", "", mention_text.upper())
    alias_upper = [re.sub(r"[^A-Z0-9+]", "", alias.upper()) for alias in candidate.aliases]
    if mention_upper and mention_upper in alias_upper:
        return 0.8
    return 0.0


def discover_candidates(
    mention: dict,
    candidates: Dict[str, CandidateEntity],
    max_candidates: int = 12,
) -> List[CandidateEntity]:
    text = normalize_text(mention.get("text"))
    category = normalize_text(mention.get("category"))
    scored: List[Tuple[float, CandidateEntity]] = []
    for candidate in candidates.values():
        score = surface_match_score(text, candidate)
        if candidate.category == category:
            score += 0.35
        if score > 0:
            scored.append((score, candidate))

    if not scored:
        for candidate in candidates.values():
            if candidate.category == category:
                scored.append((0.15, candidate))

    scored.sort(key=lambda item: (item[0], item[1].popularity), reverse=True)
    return [candidate for _, candidate in scored[:max_candidates]]


def cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def local_link_scores(
    mention: dict,
    mention_candidates: Sequence[CandidateEntity],
    vectorizer: TfidfVectorizer,
    mention_prior: Dict[Tuple[str, str], int],
    max_popularity: int,
) -> List[dict]:
    text = normalize_text(mention.get("text"))
    category = normalize_text(mention.get("category"))
    context = normalize_text(mention.get("context"))
    mention_vec = vectorizer.transform(f"{text} {category} {context}")
    total_surface_count = sum(
        count for (surface, _), count in mention_prior.items() if surface == normalize_key(text)
    )

    scored: List[dict] = []
    for candidate in mention_candidates:
        profile_text = " ".join(
            [candidate.canonical_name, candidate.category]
            + candidate.aliases
            + [counter_to_text(candidate.profile_terms)]
        )
        profile_vec = vectorizer.transform(profile_text)
        prior = 0.0
        if total_surface_count:
            prior = mention_prior.get((normalize_key(text), candidate.entity_id), 0) / total_surface_count
        surface = surface_match_score(text, candidate)
        bow = cosine(mention_vec, profile_vec)
        category_score = 1.0 if category == candidate.category else 0.15
        popularity = math.log(candidate.popularity + 1) / math.log(max_popularity + 1)
        acronym = 1.0 if surface >= 0.8 and re.fullmatch(r"[A-Za-z0-9+]+", text or "") else 0.0

        score = (
            0.30 * bow
            + 0.24 * surface
            + 0.18 * category_score
            + 0.16 * prior
            + 0.08 * popularity
            + 0.04 * acronym
        )
        scored.append(
            {
                "candidate": candidate,
                "local_score": score,
                "features": {
                    "bow_tfidf": round(bow, 6),
                    "surface_alias": round(surface, 6),
                    "category": round(category_score, 6),
                    "prior_probability": round(prior, 6),
                    "popularity": round(popularity, 6),
                    "acronym": round(acronym, 6),
                },
            }
        )
    scored.sort(key=lambda item: item["local_score"], reverse=True)
    return scored


def add_coherence_scores(scored_mentions: List[List[dict]], window_size: int = 4) -> None:
    top_candidates = [
        scored[0]["candidate"] if scored else None
        for scored in scored_mentions
    ]
    for index, scored in enumerate(scored_mentions):
        neighbors = [
            candidate
            for j, candidate in enumerate(top_candidates)
            if candidate is not None and 0 < abs(j - index) <= window_size
        ]
        for item in scored:
            candidate = item["candidate"]
            coherence_values = []
            for neighbor in neighbors:
                same_category = 1.0 if candidate.category == neighbor.category else 0.0
                term_overlap = jaccard(candidate.profile_terms.keys(), neighbor.profile_terms.keys())
                coherence_values.append(0.65 * same_category + 0.35 * term_overlap)
            coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
            item["coherence_score"] = coherence
            item["final_score"] = 0.78 * item["local_score"] + 0.22 * coherence
        scored.sort(key=lambda item: item["final_score"], reverse=True)


def enforce_cluster_consistency(scored_mentions: List[List[dict]], entities: Sequence[dict]) -> None:
    clusters: Dict[str, Counter] = defaultdict(Counter)
    for mention, scored in zip(entities, scored_mentions):
        if scored:
            clusters[normalize_key(mention.get("text"))][scored[0]["candidate"].entity_id] += 1

    majority: Dict[str, str] = {}
    for surface, votes in clusters.items():
        if votes:
            majority[surface] = votes.most_common(1)[0][0]

    for mention, scored in zip(entities, scored_mentions):
        preferred = majority.get(normalize_key(mention.get("text")))
        if not preferred:
            continue
        for item in scored:
            if item["candidate"].entity_id == preferred:
                item["final_score"] = min(item.get("final_score", item["local_score"]) + 0.04, 1.0)
        scored.sort(key=lambda item: item.get("final_score", item["local_score"]), reverse=True)


def disambiguate_entities(
    entities: Sequence[dict],
    domain_dict: Dict[str, List[str]],
    preprocessing_stats: dict = None,
) -> Tuple[List[dict], dict]:
    preprocessing_stats = preprocessing_stats or {}
    candidates = build_knowledge_base(domain_dict, entities)
    documents = []
    documents.extend(
        " ".join(
            [
                candidate.canonical_name,
                candidate.category,
                " ".join(candidate.aliases),
                counter_to_text(candidate.profile_terms),
            ]
        )
        for candidate in candidates.values()
    )
    documents.extend(
        " ".join(
            [
                normalize_text(entity.get("text")),
                normalize_text(entity.get("category")),
                normalize_text(entity.get("context")),
            ]
        )
        for entity in entities
    )
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    mention_prior: Dict[Tuple[str, str], int] = defaultdict(int)
    for entity in entities:
        text = normalize_text(entity.get("text"))
        category = normalize_text(entity.get("category"))
        entity_id = make_entity_id(category, text)
        mention_prior[(normalize_key(text), entity_id)] += int(entity.get("mention_count", 1) or 1)

    max_popularity = max((candidate.popularity for candidate in candidates.values()), default=1)
    scored_mentions: List[List[dict]] = []
    for entity in entities:
        mention_candidates = discover_candidates(entity, candidates)
        scored_mentions.append(
            local_link_scores(entity, mention_candidates, vectorizer, mention_prior, max_popularity)
        )

    add_coherence_scores(scored_mentions)
    enforce_cluster_consistency(scored_mentions, entities)

    disambiguated: List[dict] = []
    linked_count = 0
    for entity, scored in zip(entities, scored_mentions):
        item = dict(entity)
        if scored:
            best = scored[0]
            candidate: CandidateEntity = best["candidate"]
            linked_count += 1
            item.update(
                {
                    "linked_entity_id": candidate.entity_id,
                    "canonical_name": candidate.canonical_name,
                    "disambiguated_category": candidate.category,
                    "disambiguation_score": round(best.get("final_score", best["local_score"]), 6),
                    "disambiguation_method": (
                        "candidate_generation(dictionary/category) + "
                        "local_linking(tfidf_bow,prior,popularity,category,acronym) + "
                        "collaborative_graph_coherence + mention_clustering"
                    ),
                    "features": best["features"],
                    "candidate_ranking": serialize_ranking(scored[:5]),
                }
            )
        else:
            item.update(
                {
                    "linked_entity_id": "",
                    "canonical_name": normalize_text(entity.get("text")),
                    "disambiguated_category": normalize_text(entity.get("category")),
                    "disambiguation_score": 0.0,
                    "disambiguation_method": "unlinked",
                    "features": {},
                    "candidate_ranking": [],
                }
            )
        disambiguated.append(item)

    stats = {
        "entity_count": len(entities),
        "preprocessing": preprocessing_stats,
        "candidate_count": len(candidates),
        "linked_count": linked_count,
        "method_components": [
            "候选实体发现：领域词典、抽取类别、别名/缩略语扩展",
            "候选实体链接：TF-IDF BOW 上下文相似度、先验概率、类别特征、实体流行度、缩略语特征",
            "协同实体链接：文档窗口内候选实体图一致性重排",
            "基于聚类的消歧：同名 mention 聚类一致化",
        ],
    }
    return disambiguated, stats


def serialize_ranking(scored: Sequence[dict]) -> List[dict]:
    ranking = []
    for item in scored:
        candidate: CandidateEntity = item["candidate"]
        ranking.append(
            {
                "entity_id": candidate.entity_id,
                "canonical_name": candidate.canonical_name,
                "category": candidate.category,
                "local_score": round(item["local_score"], 6),
                "coherence_score": round(item.get("coherence_score", 0.0), 6),
                "final_score": round(item.get("final_score", item["local_score"]), 6),
            }
        )
    return ranking


def save_json(path: Path, source: dict, stats: dict, entities: Sequence[dict]) -> None:
    output = {
        "source_file": source.get("source_file", ""),
        "text_length": source.get("text_length"),
        "entity_count": len(entities),
        "disambiguation": stats,
        "entities": list(entities),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, entities: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "实体名称",
                "抽取类别",
                "规范实体",
                "消歧类别",
                "实体ID",
                "消歧得分",
                "起始位置",
                "结束位置",
                "上下文",
            ]
        )
        for entity in entities:
            writer.writerow(
                [
                    entity.get("text", ""),
                    entity.get("category", ""),
                    entity.get("canonical_name", ""),
                    entity.get("disambiguated_category", ""),
                    entity.get("linked_entity_id", ""),
                    entity.get("disambiguation_score", ""),
                    entity.get("start", ""),
                    entity.get("end", ""),
                    entity.get("context", ""),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity disambiguation after entity extraction")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="实体抽取结果 JSON 路径")
    parser.add_argument("--dict", default=DEFAULT_DICT, help="领域词典 JSON 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="消歧结果 JSON 输出路径")
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="消歧结果 CSV 输出路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    dict_path = Path(args.dict)
    output_path = Path(args.output)
    csv_path = Path(args.csv_output)

    source, entities = load_extraction_result(input_path)
    cleaned_entities, preprocessing_stats = preprocess_entities(entities)
    domain_dict = load_domain_dict(dict_path, cleaned_entities)
    disambiguated, stats = disambiguate_entities(cleaned_entities, domain_dict, preprocessing_stats)
    save_json(output_path, source, stats, disambiguated)
    save_csv(csv_path, disambiguated)

    print(f"输入文件: {input_path}")
    print(f"实体数量: {len(entities)}")
    print(f"去除嵌套小实体: {preprocessing_stats['nested_entity_removed_count']}")
    print(f"去除重复实体: {preprocessing_stats['duplicate_entity_removed_count']}")
    print(f"消歧输出实体: {len(disambiguated)}")
    print(f"候选实体数量: {stats['candidate_count']}")
    print(f"成功链接数量: {stats['linked_count']}")
    print(f"结果文件: {output_path}")
    print(f"CSV明细: {csv_path}")


if __name__ == "__main__":
    main()
