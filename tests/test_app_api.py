import asyncio
import os
import unittest
from unittest.mock import patch


try:
    import app
except ImportError as exc:  # FastAPI may not be installed before requirements are applied.
    app = None
    APP_IMPORT_ERROR = exc
else:
    APP_IMPORT_ERROR = None


class AppApiTest(unittest.TestCase):
    def test_missing_dashscope_key_returns_clear_answer_and_matches(self) -> None:
        if APP_IMPORT_ERROR is not None:
            self.skipTest(f"app dependencies are not installed: {APP_IMPORT_ERROR}")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False):
            response = asyncio.run(app.ask(app.AskRequest(question="数据结构分为哪些类型？")))

        self.assertIn("DASHSCOPE_API_KEY", response.error)
        self.assertIn("未配置", response.answer)
        self.assertTrue(response.matches)


if __name__ == "__main__":
    unittest.main()
