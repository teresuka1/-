import unittest

from kg_search import KnowledgeGraphSearch


class KnowledgeGraphSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.search = KnowledgeGraphSearch()

    def test_data_structure_type_question_matches_logical_and_physical_structure(self) -> None:
        matches = self.search.search("数据结构分为哪些类型？", limit=10)
        triples = {(match.subject, match.predicate, match.object) for match in matches}

        self.assertIn(("数据结构", "包含", "逻辑结构"), triples)
        self.assertIn(("数据结构", "包含", "物理结构"), triples)

    def test_linked_list_operation_question_matches_supported_operations(self) -> None:
        matches = self.search.search("链表支持哪些操作？", limit=10)
        triples = {(match.subject, match.predicate, match.object) for match in matches}

        self.assertIn(("链表", "支持操作", "插入"), triples)
        self.assertIn(("链表", "支持操作", "删除"), triples)

    def test_prim_question_matches_minimum_spanning_tree_relation(self) -> None:
        matches = self.search.search("Prim算法用于什么？", limit=10)

        self.assertTrue(
            any(
                "Prim算法" in (match.subject, match.object)
                and "最小生成树" in (match.subject, match.object)
                for match in matches
            )
        )


if __name__ == "__main__":
    unittest.main()
