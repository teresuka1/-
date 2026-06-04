import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TEXT_INPUT = "data/ds.txt"
DEFAULT_ENTITY_INPUT = "data/ds_entities_disambiguated.json"
DEFAULT_OUTPUT = "data/ds_relations.json"
DEFAULT_CSV_OUTPUT = "data/ds_relations.csv"

# 这些类别集合用于关系匹配时的轻量级类型约束，
# 让规则不只适用于单一文本，而能迁移到相似教材语料。
STRUCTURE_CATEGORIES = {
    "基础概念",
    "逻辑结构",
    "数据结构",
    "结构子类",
}
STORAGE_CATEGORIES = {"存储结构"}
ALGORITHM_CATEGORIES = {"算法"}
OPERATION_CATEGORIES = {"操作"}
PROPERTY_CATEGORIES = {"性质特征"}
APPLICATION_CATEGORIES = {"应用问题"}


@dataclass(frozen=True)
class SentenceSpan:
    """保存句子文本及其在原文中的绝对字符位置。"""

    # 句子在原文中的起始字符下标。
    start: int
    # 句子在原文中的结束字符下标。
    end: int
    # 当前句子的文本内容。
    text: str


@dataclass(frozen=True)
class Mention:
    """表示一个已经对齐回原文位置的实体提及。"""

    # 实体在原文中的表面形式，即文本里实际出现的名称。
    mention_text: str
    # 实体消歧后的规范名称，用于统一同一实体的不同写法。
    canonical_name: str
    # 实体的唯一标识符，供关系三元组中引用。
    entity_id: str
    # 实体所属的类别，例如“算法”“数据结构”等。
    category: str
    # 该 mention 在原文中的起始字符下标。
    start: int
    # 该 mention 在原文中的结束字符下标。
    end: int
    # mention 所在的上下文句子或局部文本。
    context: str


@dataclass(frozen=True)
class PairRule:
    """表示一条由触发词和类别约束共同驱动的句级关系规则。"""

    # 规则内部名称，用于区分不同规则类型。
    name: str
    # 最终输出到知识图谱中的关系名称。
    predicate: str
    # 头实体与尾实体之间应出现的左侧触发词集合。
    left_triggers: Tuple[str, ...]
    # 尾实体之后可选的补充触发词集合。
    right_triggers: Tuple[str, ...] = ()
    # 当前规则允许作为头实体的类别集合。
    subject_categories: Tuple[str, ...] = ()
    # 当前规则允许作为尾实体的类别集合。
    object_categories: Tuple[str, ...] = ()
    # 头尾实体之间允许的最大字符距离，用于限制局部匹配范围。
    max_char_gap: int = 120
    # 当前规则匹配成功后赋予关系的默认置信度。
    confidence: float = 0.8


# 这里定义实体对关系抽取所使用的规则集合。
# 当前版本只保留两类最终输出关系：
# 1）属于：表示类型归属或上下位关系；
# 2）包含：统一表示列举、成员、部分构成等较宽泛的包含关系。
# 其中“由……组成/构成”仍然保留独立匹配规则，但最终输出关系统一记为“包含”。
PAIR_RULES: Tuple[PairRule, ...] = (
    PairRule(
        name="hypernym",
        predicate="属于",
        left_triggers=("是一种", "是一类", "属于"),
        subject_categories=tuple(STRUCTURE_CATEGORIES | ALGORITHM_CATEGORIES),
        object_categories=tuple(
            STRUCTURE_CATEGORIES | STORAGE_CATEGORIES | ALGORITHM_CATEGORIES | APPLICATION_CATEGORIES
        ),
        max_char_gap=80,
        confidence=0.92,
    ),
    PairRule(
        name="contains",
        predicate="包含",
        left_triggers=("包括", "包含", "分为", "分成", "可分为"),
        subject_categories=tuple(
            STRUCTURE_CATEGORIES | STORAGE_CATEGORIES | ALGORITHM_CATEGORIES | APPLICATION_CATEGORIES
        ),
        object_categories=tuple(
            STRUCTURE_CATEGORIES
            | STORAGE_CATEGORIES
            | ALGORITHM_CATEGORIES
            | OPERATION_CATEGORIES
            | PROPERTY_CATEGORIES
            | APPLICATION_CATEGORIES
        ),
        max_char_gap=120,
        confidence=0.9,
    ),
    PairRule(
        name="composition_contains",
        predicate="包含",
        left_triggers=("由",),
        right_triggers=("组成", "构成"),
        subject_categories=tuple(STRUCTURE_CATEGORIES | STORAGE_CATEGORIES),
        object_categories=tuple(STRUCTURE_CATEGORIES | PROPERTY_CATEGORIES | STORAGE_CATEGORIES),
        max_char_gap=100,
        confidence=0.9,
    ),
)


def normalize_text(text: object) -> str:
    """统一处理 BOM 和换行符，同时尽量保持原始文本内容不变。"""

    value = str(text or "").replace("\ufeff", "")
    return re.sub(r"\r\n?", "\n", value)


def normalize_key(text: object) -> str:
    """构造用于比较的键，忽略空白差异并统一大小写。"""

    return re.sub(r"\s+", "", normalize_text(text)).strip().lower()


def safe_json_load(path: Path, default: object) -> object:
    """安全读取 JSON，避免格式异常时导致整个脚本直接崩溃。"""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def iter_sentences(text: str) -> Iterable[SentenceSpan]:
    """把原文切分成句子片段，并保留每个句子在原文中的绝对位置。

    这里同时把全角分号和半角分号都当作句边界，
    因为课程笔记或教材内容里可能会混用中英文标点。
    """

    for match in re.finditer(r".+?(?:[。！？；;\n]|$)", text, flags=re.S):
        raw = match.group()
        left_trimmed = len(raw) - len(raw.lstrip())
        right_trimmed = len(raw) - len(raw.rstrip())
        start = match.start() + left_trimmed
        end = match.end() - right_trimmed
        sentence = raw.strip()
        if sentence and end > start:
            yield SentenceSpan(start, end, sentence)


def make_entity_id(category: str, name: str) -> str:
    """在消歧结果缺少实体 ID 时，生成一个兜底用的实体 ID。"""

    raw = f"{category}:{name}"
    return re.sub(r"\s+", "_", raw)


def load_mentions(entity_path: Path) -> List[Mention]:
    """把去重后的实体结果重新展开到 mention 级别。

    实体消歧结果里通常保存的是合并后的实体，以及可选的 `occurrences`。
    但关系抽取依赖句内共现，因此需要恢复到“每次出现”的粒度。
    """

    payload = safe_json_load(entity_path, {"entities": []})
    entities = payload.get("entities", []) if isinstance(payload, dict) else []
    mentions: List[Mention] = []

    for entity in entities:
        # 优先使用消歧后的字段；若不存在，则回退到实体抽取阶段字段，
        # 这样脚本在较简化的实体 JSON 上也仍然可用。
        canonical_name = normalize_text(entity.get("canonical_name") or entity.get("text"))
        category = normalize_text(entity.get("disambiguated_category") or entity.get("category"))
        entity_id = normalize_text(entity.get("linked_entity_id")) or make_entity_id(category, canonical_name)
        occurrences = entity.get("occurrences")

        if isinstance(occurrences, list) and occurrences:
            for occ in occurrences:
                mentions.append(
                    Mention(
                        mention_text=normalize_text(occ.get("text") or entity.get("text") or canonical_name),
                        canonical_name=canonical_name,
                        entity_id=entity_id,
                        category=category,
                        start=int(occ.get("start", entity.get("start", -1))),
                        end=int(occ.get("end", entity.get("end", -1))),
                        context=normalize_text(occ.get("context") or entity.get("context")),
                    )
                )
            continue

        # 如果没有 occurrences，就把当前实体本身当作一次 mention，
        # 保证脚本面对更简单的输入格式时也能运行。
        mentions.append(
            Mention(
                mention_text=normalize_text(entity.get("text") or canonical_name),
                canonical_name=canonical_name,
                entity_id=entity_id,
                category=category,
                start=int(entity.get("start", -1)),
                end=int(entity.get("end", -1)),
                context=normalize_text(entity.get("context")),
            )
        )

    mentions = [
        mention
        for mention in mentions
        if mention.canonical_name and mention.category and mention.start >= 0 and mention.end > mention.start
    ]
    mentions.sort(key=lambda item: (item.start, item.end, item.canonical_name))
    return mentions


def category_allowed(category: str, allowed: Sequence[str]) -> bool:
    """判断某个类别是否满足当前规则的类别约束。"""

    if not allowed:
        return True
    return category in set(allowed)


def sentence_mentions(sentence: SentenceSpan, mentions: Sequence[Mention]) -> List[Mention]:
    """收集完全落在当前句子范围内的所有 mention。"""

    selected = [
        mention
        for mention in mentions
        if sentence.start <= mention.start and mention.end <= sentence.end
    ]

    # 句内匹配前先做一次最长优先去嵌套，避免把“连通图”中的“连通”、
    # “生成树”中的“树”这类短 mention 一并带入关系抽取。
    selected.sort(
        key=lambda item: (item.start, -(item.end - item.start), item.canonical_name)
    )
    filtered: List[Mention] = []
    for mention in selected:
        if any(
            kept.start <= mention.start and mention.end <= kept.end
            for kept in filtered
        ):
            continue
        filtered = [
            kept
            for kept in filtered
            if not (mention.start <= kept.start and kept.end <= mention.end)
        ]
        filtered.append(mention)

    filtered.sort(key=lambda item: (item.start, item.end, item.canonical_name))
    return filtered


def find_trigger(text: str, triggers: Sequence[str]) -> Optional[str]:
    """选择匹配到的最长触发词，减少短字符串带来的歧义。"""

    if not triggers:
        return None
    matched = [trigger for trigger in triggers if trigger and trigger in text]
    if not matched:
        return None
    matched.sort(key=len, reverse=True)
    return matched[0]


def is_name_character(char: str) -> bool:
    """判断字符是否像术语内部字符，而不是分隔符。"""

    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9+]", char))


def match_rule(rule: PairRule, sentence: SentenceSpan, subject: Mention, obj: Mention) -> Optional[dict]:
    """检查一个实体对在当前句子中是否满足某条关系规则。"""
    # 排除自环关系，并强制按从左到右的顺序匹配，简化规则模板。
    if subject.entity_id == obj.entity_id:
        return None
    if subject.start >= obj.start:
        return None

    # 先用类别约束过滤，尽早排除明显不合理的实体对。
    if not category_allowed(subject.category, rule.subject_categories):
        return None
    if not category_allowed(obj.category, rule.object_categories):
        return None

    # 只检查两个实体之间的文本以及尾实体之后的局部后缀，
    # 保证规则始终停留在句内，避免跨子句误匹配。
    local_subject_end = subject.end - sentence.start
    local_object_start = obj.start - sentence.start
    local_object_end = obj.end - sentence.start
    between_text = sentence.text[local_subject_end:local_object_start]
    suffix_text = sentence.text[local_object_end:]
    previous_char = sentence.text[local_object_start - 1] if local_object_start > 0 else ""
    next_char = sentence.text[local_object_end] if local_object_end < len(sentence.text) else ""

    if len(between_text) > rule.max_char_gap:
        return None

    left_trigger = find_trigger(between_text, rule.left_triggers)
    if not left_trigger:
        return None

    right_trigger = find_trigger(suffix_text, rule.right_triggers) if rule.right_triggers else None
    if rule.right_triggers and not right_trigger:
        return None

    # 当前只保留两类关系，规则过滤也相应精简。
    # “由……组成/构成”仍走独立规则，但最终统一映射为“包含”。
    if rule.name == "contains":
        # “包含指向后继结点的指针”中真正被包含的是“指针”，
        # 不应把“后继结点/前驱结点”误识别为被包含对象。
        if "指向" in between_text and not any(
            keyword in obj.canonical_name for keyword in ("指针", "引用")
        ):
            return None

        # “连通图”中的“连通”、“生成树”中的“树”这类短实体，
        # 实际上嵌在更长术语内部，不应被当成“包括”列表项。
        if not any(keyword in obj.canonical_name for keyword in ("指针", "引用")):
            if next_char and is_name_character(next_char):
                return None
            if (
                previous_char
                and is_name_character(previous_char)
                and not between_text.rstrip().endswith(left_trigger)
            ):
                return None

    if rule.name == "hypernym":
        # “A 是一种用于查找数据元素的数据结构”这类定义句中，
        # 只保留真正落在句末的上位类，避免把中间名词误识别为尾实体。
        trailing_text = re.sub(r"[。！？；;，、,\s]+", "", suffix_text)
        if trailing_text:
            return None

    # 返回结果中直接保留证据句，方便后续人工检查和导出为图谱关系文件。
    return {
        "subject_id": subject.entity_id,
        "subject": subject.canonical_name,
        "subject_category": subject.category,
        "predicate": rule.predicate,
        "object_id": obj.entity_id,
        "object": obj.canonical_name,
        "object_category": obj.category,
        "trigger": right_trigger or left_trigger,
        "rule": rule.name,
        "confidence": rule.confidence,
        "evidence": sentence.text,
        "evidence_start": sentence.start,
        "evidence_end": sentence.end,
    }


def extract_pair_relations(sentence: SentenceSpan, mentions: Sequence[Mention]) -> List[dict]:
    """枚举句内有序实体对，并依次尝试所有实体对规则。"""

    results: List[dict] = []
    for index, subject in enumerate(mentions):
        for obj in mentions[index + 1 :]:
            for rule in PAIR_RULES:
                match = match_rule(rule, sentence, subject, obj)
                if match:
                    results.append(match)
    return results

def extract_sentence_relations(sentence: SentenceSpan, mentions: Sequence[Mention]) -> List[dict]:
    """抽取当前句子中所有可识别的关系。"""

    relations: List[dict] = []
    relations.extend(extract_pair_relations(sentence, mentions))
    return relations


def deduplicate_relations(relations: Sequence[dict]) -> List[dict]:
    """对同一句证据下的重复关系去重，并保留置信度最高的一条。"""

    best: Dict[Tuple[str, str, str, str], dict] = {}
    for relation in relations:
        key = (
            relation["subject_id"],
            relation["predicate"],
            relation["object_id"],
            relation["evidence"],
        )
        current = best.get(key)
        if current is None or relation["confidence"] > current["confidence"]:
            best[key] = relation
    deduped = list(best.values())
    deduped.sort(key=lambda item: (item["subject"], item["predicate"], item["object"], item["evidence_start"]))
    return deduped


def aggregate_relations(relations: Sequence[dict]) -> List[dict]:
    """把句子级关系合并成图谱级关系记录。"""

    grouped: Dict[Tuple[str, str, str], dict] = {}
    for relation in relations:
        key = (relation["subject_id"], relation["predicate"], relation["object_id"])
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "subject_id": relation["subject_id"],
                "subject": relation["subject"],
                "subject_category": relation["subject_category"],
                "predicate": relation["predicate"],
                "object_id": relation["object_id"],
                "object": relation["object"],
                "object_category": relation["object_category"],
                "confidence": relation["confidence"],
                "rules": [relation["rule"]],
                "triggers": [relation["trigger"]],
                "evidence_count": 0,
                "evidences": [],
            }
            grouped[key] = bucket

        bucket["confidence"] = max(bucket["confidence"], relation["confidence"])
        if relation["rule"] not in bucket["rules"]:
            bucket["rules"].append(relation["rule"])
        if relation["trigger"] not in bucket["triggers"]:
            bucket["triggers"].append(relation["trigger"])

        bucket["evidence_count"] += 1
        bucket["evidences"].append(
            {
                "text": relation["evidence"],
                "start": relation["evidence_start"],
                "end": relation["evidence_end"],
            }
        )

    aggregated = list(grouped.values())
    aggregated.sort(
        key=lambda item: (
            item["subject_category"],
            item["subject"],
            item["predicate"],
            item["object_category"],
            item["object"],
        )
    )
    return aggregated


def extract_relations(text: str, mentions: Sequence[Mention]) -> Tuple[List[dict], dict]:
    """执行完整的关系抽取流程，从句子级匹配到最终三元组聚合。"""

    sentences = list(iter_sentences(text))
    sentence_relation_candidates: List[dict] = []

    for sentence in sentences:
        local_mentions = sentence_mentions(sentence, mentions)
        if len(local_mentions) < 2:
            continue
        sentence_relation_candidates.extend(extract_sentence_relations(sentence, local_mentions))

    unique_relations = deduplicate_relations(sentence_relation_candidates)
    aggregated_relations = aggregate_relations(unique_relations)
    stats = {
        "mention_count": len(mentions),
        "sentence_count": len(sentences),
        "raw_relation_candidate_count": len(sentence_relation_candidates),
        "unique_sentence_relation_count": len(unique_relations),
        "aggregated_relation_count": len(aggregated_relations),
        "method": (
            "限定域关系抽取：基于实体消歧结果的句内实体对遍历，结合"
            "关系触发词、句式模板和实体类别约束抽取三元组"
        ),
        "relation_rules": [
            {
                "name": rule.name,
                "predicate": rule.predicate,
                "left_triggers": list(rule.left_triggers),
                "right_triggers": list(rule.right_triggers),
                "subject_categories": list(rule.subject_categories),
                "object_categories": list(rule.object_categories),
            }
            for rule in PAIR_RULES
        ],
    }
    return aggregated_relations, stats


def save_json(output_path: Path, text_path: Path, entity_path: Path, relations: Sequence[dict], stats: dict) -> None:
    """保存包含方法信息的结构化 JSON 结果。"""

    output = {
        "source_file": str(text_path),
        "entity_source_file": str(entity_path),
        "relation_count": len(relations),
        "relation_extraction": stats,
        "relations": list(relations),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(csv_path: Path, relations: Sequence[dict]) -> None:
    """保存便于人工查看的扁平化 CSV 结果。"""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "头实体",
                "头实体类别",
                "关系",
                "尾实体",
                "尾实体类别",
                "置信度",
                "触发词",
                "证据数量",
                "示例证据",
            ]
        )
        for relation in relations:
            writer.writerow(
                [
                    relation["subject"],
                    relation["subject_category"],
                    relation["predicate"],
                    relation["object"],
                    relation["object_category"],
                    relation["confidence"],
                    " / ".join(relation["triggers"]),
                    relation["evidence_count"],
                    relation["evidences"][0]["text"] if relation["evidences"] else "",
                ]
            )


def validate_inputs(text_path: Path, entity_path: Path) -> None:
    """在输入文件缺失时立即报错，并给出清晰提示。"""

    if not text_path.exists():
        raise FileNotFoundError(f"原始文本文件不存在: {text_path}")
    if not entity_path.exists():
        raise FileNotFoundError(f"实体消歧结果文件不存在: {entity_path}")


def main() -> None:
    """解析参数、校验输入、执行抽取并写出结果。"""

    parser = argparse.ArgumentParser(description="基于原始文本和实体消歧结果的限定域关系抽取")
    parser.add_argument("--input", default=DEFAULT_TEXT_INPUT, help="原始文本路径")
    parser.add_argument("--entities", default=DEFAULT_ENTITY_INPUT, help="实体消歧结果 JSON 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="关系抽取 JSON 输出路径")
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="关系抽取 CSV 输出路径")
    args = parser.parse_args()

    text_path = Path(args.input)
    entity_path = Path(args.entities)
    output_path = Path(args.output)
    csv_path = Path(args.csv_output)

    validate_inputs(text_path, entity_path)
    text = normalize_text(text_path.read_text(encoding="utf-8"))
    mentions = load_mentions(entity_path)
    relations, stats = extract_relations(text, mentions)
    save_json(output_path, text_path, entity_path, relations, stats)
    save_csv(csv_path, relations)

    print(f"原始文本: {text_path}")
    print(f"实体结果: {entity_path}")
    print(f"mention 数量: {len(mentions)}")
    print(f"聚合关系数: {len(relations)}")
    print(f"JSON 输出: {output_path}")
    print(f"CSV 输出: {csv_path}")


if __name__ == "__main__":
    main()
