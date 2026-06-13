import re
import unittest

from relation_extraction import Mention, extract_relations


def make_mention(
    text: str,
    surface: str,
    occurrence_index: int,
    canonical_name: str,
    entity_id: str,
    category: str,
) -> Mention:
    matches = list(re.finditer(re.escape(surface), text))
    match = matches[occurrence_index]
    return Mention(
        mention_text=surface,
        canonical_name=canonical_name,
        entity_id=entity_id,
        category=category,
        start=match.start(),
        end=match.end(),
        context=text,
    )


class RelationExtractionTest(unittest.TestCase):
    def test_extract_relations_captures_storage_membership_and_operations(self) -> None:
        text = "链表是一种线性表。链表采用链式存储结构。链表支持插入和删除操作。"
        mentions = [
            make_mention(text, "链表", 0, "链表", "结构子类:链表", "结构子类"),
            make_mention(text, "线性表", 0, "线性表", "数据结构:线性表", "数据结构"),
            make_mention(text, "链表", 1, "链表", "结构子类:链表", "结构子类"),
            make_mention(text, "链式存储结构", 0, "链式存储结构", "存储结构:链式存储结构", "存储结构"),
            make_mention(text, "链表", 2, "链表", "结构子类:链表", "结构子类"),
            make_mention(text, "插入", 0, "插入", "操作:插入", "操作"),
            make_mention(text, "删除", 0, "删除", "操作:删除", "操作"),
        ]

        relations, stats = extract_relations(text, mentions)
        triples = {(item["subject"], item["predicate"], item["object"]) for item in relations}

        self.assertIn(("链表", "采用存储结构", "链式存储结构"), triples)
        self.assertIn(("链表", "属于", "线性表"), triples)
        self.assertIn(("链表", "支持操作", "插入"), triples)
        self.assertIn(("链表", "支持操作", "删除"), triples)
        self.assertEqual(stats["aggregated_relation_count"], 4)

    def test_problem_to_algorithm_list_uses_qiujie_predicate_not_contains(self) -> None:
        text = "最小生成树的算法包括Prim算法和Kruskal算法。"
        mentions = [
            make_mention(text, "最小生成树", 0, "最小生成树", "应用问题:最小生成树", "应用问题"),
            make_mention(text, "Prim算法", 0, "Prim算法", "算法:Prim算法", "算法"),
            make_mention(text, "Kruskal算法", 0, "Kruskal算法", "算法:Kruskal算法", "算法"),
        ]

        relations, _ = extract_relations(text, mentions)
        triples = {(item["subject"], item["predicate"], item["object"]) for item in relations}

        self.assertIn(("最小生成树", "求解问题", "Prim算法"), triples)
        self.assertIn(("最小生成树", "求解问题", "Kruskal算法"), triples)
        self.assertNotIn(("最小生成树", "包含", "Prim算法"), triples)
        self.assertNotIn(("最小生成树", "包含", "Kruskal算法"), triples)

    def test_repeated_identical_evidence_sentences_keep_two_evidence_items(self) -> None:
        text = "链表支持插入操作。链表支持插入操作。"
        mentions = [
            make_mention(text, "链表", 0, "链表", "结构子类:链表", "结构子类"),
            make_mention(text, "插入", 0, "插入", "操作:插入", "操作"),
            make_mention(text, "链表", 1, "链表", "结构子类:链表", "结构子类"),
            make_mention(text, "插入", 1, "插入", "操作:插入", "操作"),
        ]

        relations, stats = extract_relations(text, mentions)

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["evidence_count"], 2)
        self.assertEqual(len(relations[0]["evidences"]), 2)
        self.assertEqual(stats["unique_sentence_relation_count"], 2)
        self.assertEqual(stats["aggregated_relation_count"], 1)


if __name__ == "__main__":
    unittest.main()
