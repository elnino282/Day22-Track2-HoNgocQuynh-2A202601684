"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import hashlib
import json
import math
import shutil
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

# RAGAS/tqdm writes progress to stderr. Windows PowerShell 5.1 wraps any native
# stderr line as a scary-looking NativeCommandError even when Python is healthy.
# Merge it into the already UTF-8-configured stdout stream so logs remain clean
# and genuine tracebacks are still captured in full by run_utf8.ps1.
sys.stderr = sys.stdout

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from utils.ragas_compat import install_vertexai_import_shim

install_vertexai_import_shim()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS
from prompt_versions import PROMPT_V1, PROMPT_V2, SYSTEM_V1, SYSTEM_V2


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Imported from prompt_versions.py to guarantee the evaluated prompts are
# byte-for-byte identical to the versions pushed in step 2.
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]

# RAGAS defaults answer relevancy to strictness=3, which asks the evaluator
# for n=3 generations in one API call. OpenAI Responses-compatible proxies
# commonly support n=1 only. Strictness 1 still computes the rubric's standard
# answer_relevancy metric, using one generated reverse-question per answer.
ANSWER_RELEVANCY_METRIC = AnswerRelevancy(strictness=1)
CACHE_PATH = Path(__file__).parent.parent / "data" / "ragas_inputs_cache.json"


def _cache_fingerprint() -> str:
    """Identify the exact model, prompts, questions, KB and retrieval setup."""
    payload = {
        "provider": config.PROVIDER,
        "openai_model": config.OPENAI_MODEL,
        "gemini_model": config.GEMINI_MODEL,
        "anthropic_model": config.ANTHROPIC_MODEL,
        "ollama_model": config.OLLAMA_MODEL,
        "openrouter_model": config.OPENROUTER_MODEL,
        "system_v1": SYSTEM_V1,
        "system_v2": SYSTEM_V2,
        "qa_pairs": QA_PAIRS,
        "knowledge_base": load_knowledge_base(),
        "chunk_size": 500,
        "chunk_overlap": 50,
        "retriever_k": 3,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_rag_outputs():
    """Load complete compatible V1/V2 inputs from a previous interrupted run."""
    if not CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cache.get("fingerprint") != _cache_fingerprint():
            print("ℹ️  Bỏ qua RAGAS cache cũ vì model/prompt/dữ liệu đã thay đổi.")
            return None
        v1_results = cache["v1_results"]
        v2_results = cache["v2_results"]
        if len(v1_results) != 50 or len(v2_results) != 50:
            return None
        if not all(isinstance(row.get("contexts"), list) for row in v1_results + v2_results):
            return None
        print(f"♻️  Đã tải lại 100 RAG outputs từ {CACHE_PATH}")
        return v1_results, v2_results
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"⚠️  Không thể dùng RAGAS cache, sẽ tạo lại: {exc}")
        return None


def save_rag_outputs_cache(v1_results: list, v2_results: list) -> None:
    """Atomically checkpoint expensive RAG outputs before metric evaluation."""
    cache = {
        "fingerprint": _cache_fingerprint(),
        "provider": config.PROVIDER,
        "model": config.OPENAI_MODEL if config.PROVIDER == "openai" else config.PROVIDER,
        "sample_count_per_version": len(QA_PAIRS),
        "v1_results": v1_results,
        "v2_results": v2_results,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(CACHE_PATH)
    print(f"💾 Đã checkpoint 100 RAG outputs vào {CACHE_PATH}")


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    def process(qa):
        out = run_rag(retriever, llm, prompt, qa["question"])
        return {
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        }

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        results = list(executor.map(process, QA_PAIRS))

    for i, qa in enumerate(QA_PAIRS, 1):
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    if len(rag_results) != len(QA_PAIRS):
        raise ValueError(
            f"RAGAS cần đủ {len(QA_PAIRS)} mẫu, nhận được {len(rag_results)}"
        )

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(
        temperature=0,
        request_timeout=config.RAGAS_TIMEOUT,
        max_retries=config.RAGAS_MAX_RETRIES,
    )
    emb_eval = get_embeddings(
        request_timeout=config.RAGAS_TIMEOUT,
        max_retries=config.RAGAS_MAX_RETRIES,
    )

    print(
        f"   Evaluator: {config.OPENAI_MODEL if config.PROVIDER == 'openai' else config.PROVIDER}"
        " | answer_relevancy strictness=1"
        f" | timeout={config.RAGAS_TIMEOUT:.0f}s"
        f" | workers={config.RAGAS_MAX_WORKERS}"
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            ANSWER_RELEVANCY_METRIC,
            context_recall,
            context_precision,
        ],
        llm=llm_eval,
        embeddings=emb_eval,
        raise_exceptions=False,
        show_progress=False,
        run_config=RunConfig(
            timeout=int(config.RAGAS_TIMEOUT),
            max_retries=config.RAGAS_MAX_RETRIES,
            max_workers=config.RAGAS_MAX_WORKERS,
            seed=42,
        ),
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in METRIC_NAMES:
        raw = result[key]
        valid = [float(v) for v in raw if v is not None and not math.isnan(float(v))]
        if not valid:
            raise RuntimeError(f"RAGAS không trả về điểm hợp lệ cho metric '{key}'")
        scores[key] = float(np.mean(valid))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    if len(QA_PAIRS) != 50:
        raise RuntimeError(f"Rubric yêu cầu đúng 50 QA pairs, hiện có {len(QA_PAIRS)}")

    cached_outputs = load_cached_rag_outputs()
    if cached_outputs is None:
        vectorstore = setup_vectorstore()
        # Thu thập kết quả RAG cho cả V1 và V2
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        save_rag_outputs_cache(v1_results, v2_results)
    else:
        v1_results, v2_results = cached_outputs

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    winner = "v1" if sum(v1_scores.values()) >= sum(v2_scores.values()) else "v2"
    analysis = (
        f"Prompt {winner.upper()} có điểm trung bình cao hơn. V1 ưu tiên câu trả lời "
        "ngắn gọn; V2 buộc câu trả lời nêu evidence và confidence. Cả hai đều "
        "cấm dùng kiến thức ngoài context để tối ưu faithfulness."
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count_per_version": len(QA_PAIRS),
        "total_rag_answers": len(v1_results) + len(v2_results),
        "metrics": METRIC_NAMES,
        "answer_relevancy_strictness": 1,
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "winner_by_mean": winner,
        "analysis": analysis,
    }
    project_root = Path(__file__).parent.parent
    report_path = project_root / "data" / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evidence_report = project_root / "evidence" / "03_ragas_report.json"
    evidence_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, evidence_report)
    print(f"💾 Đã lưu báo cáo vào {report_path}")
    print(f"💾 Đã sao chép evidence vào {evidence_report}")
    print(f"\nPhân tích: {analysis}")


if __name__ == "__main__":
    main()
