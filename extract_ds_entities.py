import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_DICT: Dict[str, List[str]] = {
    "基础概念": [
        "数据结构",
        "逻辑结构",
        "物理结构",
        "数据元素",
        "结点",
        "前驱结点",
        "后继结点",
        "节点",
        "边",
        "顶点",
        "指针",
        "关键字",
        "存储空间",
        "内存",
        "权值",
    ],
    "逻辑结构": [
        "线性结构",
        "树形结构",
        "图结构",
        "集合结构",
    ],
    "存储结构": [
        "堆分配存储",
        "块链存储",
        "顺序存储",
        "链式存储",
        "连续存储",
        "非连续存储",
    ],
    "数据结构": [
        "线性表",
        "顺序表",
        "链表",
        "栈",
        "队列",
        "串",
        "数组",
        "广义表",
        "树",
        "图",
        "查找表",
        "矩阵",
        "森林",
        "哈希表",
        "散列表",
        "堆",
    ],
    "结构子类": [
        "静态链表",
        "动态链表",
        "单链表",
        "双向链表",
        "循环链表",
        "双向循环链表",
        "顺序栈",
        "链栈",
        "顺序队列",
        "循环队列",
        "链式队列",
        "普通树",
        "二叉树",
        "线索二叉树",
        "双向线索二叉树",
        "哈夫曼树",
        "二叉排序树",
        "二叉查找树",
        "平衡二叉树",
        "红黑树",
        "稠密图",
        "稀疏图",
        "有向图",
        "无向图",
        "连通图",
        "带权连通图",
        "有向无环图",    
        "B树",
        "B+树",
        "键树",
        "静态查找表",
        "动态查找表",
        "邻接矩阵",
        "邻接表",
        "邻接多重表",
        "十字链表",
        "行逻辑链接顺序表",
        "三元组顺序表",
        "稀疏矩阵",
    ],
    "算法": [
        "BF算法",
        "KMP算法",
        "DFS",
        "BFS",
        "Prim算法",
        "Kruskal算法",
        "Dijkstra算法",
        "Floyd算法",
        "拓扑排序算法",
        "快速排序",
        "归并排序",
        "插入排序",
        "希尔排序",
        "冒泡排序",
        "选择排序",
        "堆排序",
        "基数排序",
        "顺序查找",
        "索引查找",
        "二分查找",
        "分块查找",
        "静态树表查找",
        "深度优先搜索",
        "广度优先搜索",
        "哈夫曼编码",
    ],
    "操作": [
        "插入",
        "删除",
        "查找",
        "修改",
        "遍历",
        "读取",
        "入栈",
        "出栈",
        "入队",
        "出队",
        "复制",
        "压缩存储",
        "排序",
        "匹配",
        "反转",
        "连接",
        "存储",
        "先序遍历",
        "中序遍历",
        "后序遍历",
        "层次遍历",
    ],
    "性质特征": [
        "先进先出",
        "先进后出",
        "有序",
        "无序",
        "连续存储",
        "非连续存储",
        "稳定",
        "平衡",
        "带权",
        "连通",
        "一对一",
        "一对多",
        "多对多",
    ],
    "应用问题": [
        "括号匹配",
        "表达式求值",
        "进制转换",
        "最短路径",
        "最小生成树",
        "关键路径",
        "递归调用",
        "模式匹配",
        "数据压缩",
        "编码问题",
        "数据库索引",
        "文件系统",
    ],
}


REGEX_RULES: Dict[str, List[str]] = {
    "算法": [
        r"[A-Z][A-Za-z0-9+]*算法",
        r"(深度优先|广度优先)搜索",
        r"\b[A-Z]{2,}\b",
    ],

}


def load_domain_dict(dict_path: Path) -> Dict[str, List[str]]:
    if not dict_path.exists():
        return DEFAULT_DICT
    data = json.loads(dict_path.read_text(encoding="utf-8"))
    merged = {key: list(values) for key, values in DEFAULT_DICT.items()}
    for category, terms in data.items():
        bucket = merged.setdefault(category, [])
        for term in terms:
            if term not in bucket:
                bucket.append(term)
    return merged


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    return text


def iter_sentences(text: str) -> Iterable[Tuple[int, int, str]]:
    start = 0
    for match in re.finditer(r".+?(?:[。！？；\n]|$)", text, flags=re.S):
        sentence = match.group().strip()
        if sentence:
            yield match.start(), match.end(), sentence
        start = match.end()
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            yield start, len(text), tail


def find_context(spans: List[Tuple[int, int, str]], start: int, end: int) -> str:
    for sent_start, sent_end, sentence in spans:
        if sent_start <= start and end <= sent_end:
            return sentence
    return ""


def collect_from_dict(text: str, domain_dict: Dict[str, List[str]]) -> List[dict]:
    results: List[dict] = []
    for category, terms in domain_dict.items():
        for term in sorted(set(terms), key=len, reverse=True):
            if not term.strip():
                continue
            for match in re.finditer(re.escape(term), text):
                results.append(
                    {
                        "text": match.group(),
                        "category": category,
                        "start": match.start(),
                        "end": match.end(),
                        "method": "dictionary",
                    }
                )
    return results


def collect_from_regex(text: str, regex_rules: Dict[str, List[str]]) -> List[dict]:
    results: List[dict] = []
    for category, patterns in regex_rules.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                results.append(
                    {
                        "text": match.group(),
                        "category": category,
                        "start": match.start(),
                        "end": match.end(),
                        "method": f"regex:{pattern}",
                    }
                )
    return results


def deduplicate_entities(entities: List[dict]) -> List[dict]:
    priority = {"dictionary": 2}

    def score(item: dict) -> Tuple[int, int]:
        return (priority.get(item["method"], 1), len(item["text"]))

    best = {}
    for entity in entities:
        key = (entity["text"], entity["category"], entity["start"], entity["end"])
        if key not in best or score(entity) > score(best[key]):
            best[key] = entity

    merged = list(best.values())
    merged.sort(key=lambda item: (item["start"], item["end"], item["category"]))
    return merged


def attach_context(entities: List[dict], text: str) -> List[dict]:
    spans = list(iter_sentences(text))
    for entity in entities:
        entity["context"] = find_context(spans, entity["start"], entity["end"])
        entity.pop("method", None)
    return entities


def save_entity_csv(csv_path: Path, entities: List[dict]) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["实体名称", "实体类别", "起始位置", "结束位置", "上下文"])
        for entity in entities:
            writer.writerow(
                [
                    entity["text"],
                    entity["category"],
                    entity["start"],
                    entity["end"],
                    entity["context"],
                ]
            )


def extract_entities(text: str, domain_dict: Dict[str, List[str]]) -> List[dict]:
    entities = []
    entities.extend(collect_from_dict(text, domain_dict))
    entities.extend(collect_from_regex(text, REGEX_RULES))
    entities = deduplicate_entities(entities)
    entities = attach_context(entities, text)
    return entities


def save_results(
    output_path: Path,
    csv_path: Path,
    entities: List[dict],
    source_path: Path,
    text: str,
) -> None:
    output = {
        "source_file": str(source_path),
        "text_length": len(text),
        "entity_count": len(entities),
        "entities": entities,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    save_entity_csv(csv_path, entities)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于规则和领域词典的数据结构实体抽取")
    parser.add_argument("--input", default="data/ds.txt", help="原始文本路径")
    parser.add_argument("--dict", default="data/ds_domain_dict.json", help="领域词典路径")
    parser.add_argument("--output", default="data/ds_entities.json", help="抽取结果输出路径")
    parser.add_argument("--csv-output", default="data/ds_entities.csv", help="实体明细 CSV 输出路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    dict_path = Path(args.dict)
    output_path = Path(args.output)
    csv_path = Path(args.csv_output)

    text = normalize_text(input_path.read_text(encoding="utf-8")) if input_path.exists() else ""
    domain_dict = load_domain_dict(dict_path)
    entities = extract_entities(text, domain_dict)
    save_results(output_path, csv_path, entities, input_path, text)

    print(f"输入文件: {input_path}")
    print(f"文本长度: {len(text)}")
    print(f"抽取实体数: {len(entities)}")
    print(f"结果文件: {output_path}")
    print(f"CSV明细: {csv_path}")


if __name__ == "__main__":
    main()
