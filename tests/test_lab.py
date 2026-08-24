"""Offline regression tests for rubric-critical behavior."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from guardrails import Guard, OnFailAction
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langsmith.utils import LangSmithConflictError

from qa_pairs import QA_PAIRS, SAMPLE_QUESTIONS
from prompt_versions import PROMPT_V1, PROMPT_V2, SYSTEM_V1, SYSTEM_V2


step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")
step3 = importlib.import_module("03_ragas_evaluation")
step4 = importlib.import_module("04_guardrails_validator")


class FakeRetriever:
    """Minimal retriever for unit-testing explicit retrieval functions."""

    def invoke(self, _question):
        return [Document(page_content="Grounded fact one."), Document(page_content="Fact two.")]


class LabTests(unittest.TestCase):
    def test_exactly_fifty_qa_pairs(self):
        self.assertEqual(len(SAMPLE_QUESTIONS), 50)
        self.assertEqual(len(QA_PAIRS), 50)
        self.assertEqual(SAMPLE_QUESTIONS, [row["question"] for row in QA_PAIRS])

    def test_shared_prompts_are_distinct_and_grounded(self):
        self.assertNotEqual(SYSTEM_V1, SYSTEM_V2)
        for prompt in (SYSTEM_V1, SYSTEM_V2):
            self.assertIn("{context}", prompt)
            self.assertIn("only", prompt.lower())
        self.assertIs(step3.PROMPT_V1, PROMPT_V1)
        self.assertIs(step3.PROMPT_V2, PROMPT_V2)

    def test_ab_routing_is_deterministic_and_balanced(self):
        first = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        second = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {step2.PROMPT_V1_NAME, step2.PROMPT_V2_NAME})
        with self.assertRaises(ValueError):
            step2.get_prompt_version("")

    def test_prompt_push_is_idempotent_when_content_is_unchanged(self):
        class UnchangedPromptClient:
            def push_prompt(self, *_args, **_kwargs):
                raise LangSmithConflictError(
                    "409 Nothing to commit: prompt has not changed since latest commit"
                )

        result = step2.push_prompts_to_hub(UnchangedPromptClient())
        self.assertEqual(result[step2.PROMPT_V1_NAME], "unchanged")
        self.assertEqual(result[step2.PROMPT_V2_NAME], "unchanged")

    def test_ab_query_returns_context_in_trace_output(self):
        raw_ask = getattr(step2.ask_ab, "__wrapped__", step2.ask_ab)
        result = raw_ask(
            FakeRetriever(),
            FakeListChatModel(responses=["A grounded answer."]),
            PROMPT_V1,
            "Question?",
            "v1",
        )
        self.assertEqual(result["version"], "v1")
        self.assertEqual(result["answer"], "A grounded answer.")
        self.assertEqual(result["contexts"], ["Grounded fact one.", "Fact two."])

    def test_ragas_sample_schema(self):
        rows = [
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": qa["reference"],
                "contexts": [qa["reference"]],
            }
            for qa in QA_PAIRS
        ]
        dataset = step3.build_ragas_dataset(rows)
        self.assertEqual(len(dataset), 50)
        sample = dataset[0]
        self.assertEqual(sample.user_input, rows[0]["question"])
        self.assertIsInstance(sample.retrieved_contexts, list)

    def test_ragas_proxy_uses_single_generation(self):
        self.assertEqual(step3.ANSWER_RELEVANCY_METRIC.strictness, 1)

    def test_ragas_input_cache_round_trip(self):
        rows = [
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": qa["reference"],
                "contexts": [qa["reference"]],
            }
            for qa in QA_PAIRS
        ]
        original_cache_path = step3.CACHE_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                step3.CACHE_PATH = Path(temp_dir) / "ragas_inputs_cache.json"
                step3.save_rag_outputs_cache(rows, rows)
                cached_v1, cached_v2 = step3.load_cached_rag_outputs()
                self.assertEqual(len(cached_v1), 50)
                self.assertEqual(len(cached_v2), 50)
        finally:
            step3.CACHE_PATH = original_cache_path

    def test_pii_guard_redacts_four_types(self):
        guard = Guard().use(step4.PIIDetector(on_fail=OnFailAction.FIX))
        outcome = guard.validate(
            "john@example.com (555) 867-5309 123-45-6789 4532 1234 5678 9010"
        )
        output = outcome.validated_output
        for pii_type in ("EMAIL", "PHONE", "SSN", "CREDIT_CARD"):
            self.assertIn(f"[{pii_type}_REDACTED]", output)

    def test_json_guard_repairs_and_falls_back_to_valid_json(self):
        guard = Guard().use(step4.JSONFormatter(on_fail=OnFailAction.FIX))
        repaired = guard.validate("```json\n{'name': 'Quynh',}\n```").validated_output
        self.assertEqual(json.loads(repaired), {"name": "Quynh"})

        fallback = guard.validate("not json at all").validated_output
        parsed = json.loads(fallback)
        self.assertIn("error", parsed)
        self.assertEqual(parsed["raw"], "not json at all")


if __name__ == "__main__":
    unittest.main()
