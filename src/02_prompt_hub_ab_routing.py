"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS
from prompt_versions import PROMPT_V1, PROMPT_V2, SYSTEM_V1, SYSTEM_V2


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "ho-ngoc-quynh-rag-prompt-v1"
PROMPT_V2_NAME = "ho-ngoc-quynh-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# SYSTEM_V1/SYSTEM_V2 and their templates are defined once in
# prompt_versions.py so evaluation always uses exactly the deployed prompts.


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client) -> dict[str, str]:
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    pushed = {}
    definitions = (
        (PROMPT_V1_NAME, PROMPT_V1, "V1 - concise, friendly, context-grounded"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 - structured expert answer with evidence and confidence"),
    )
    for name, prompt, description in definitions:
        try:
            pushed[name] = str(
                client.push_prompt(name, object=prompt, description=description)
            )
            print(f"✅ Đã push '{name}' → {pushed[name]}")
        except LangSmithConflictError as exc:
            # LangSmith returns HTTP 409 when an identical prompt commit already
            # exists. This is an idempotent success, not a deployment failure.
            if "nothing to commit" in str(exc).lower():
                pushed[name] = "unchanged"
                print(f"✅ Prompt '{name}' đã tồn tại và không thay đổi")
                continue
            raise RuntimeError(
                f"Conflict khi push prompt '{name}' lên Hub: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Không thể push prompt '{name}' lên Hub: {exc}") from exc
    return pushed


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Không fallback sang local: rubric yêu cầu chứng minh prompt thật sự được
    pull từ Hub, nên lỗi mạng/quyền phải làm bước này fail rõ ràng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Đã pull '{name}' từ Hub")
        except Exception as exc:
            raise RuntimeError(f"Không thể pull prompt bắt buộc '{name}' từ Hub: {exc}") from exc

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id phải là chuỗi không rỗng")
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "question": question,
        "answer": answer,
        "version": version,
        "contexts": contexts,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    def run_one(index_question):
        i, question = index_question
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        return i, request_id, question, result

    print(f"🚀 Đang xử lý với tối đa {config.MAX_WORKERS} request song song...")
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        routed_results = list(executor.map(run_one, enumerate(SAMPLE_QUESTIONS)))

    v1_count, v2_count = 0, 0
    for i, request_id, question, result in routed_results:
        version_tag = result["version"]
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}/50] [{request_id}] [prompt-{result['version']}] {question[:55]}...")
        print(f"          A: {result['answer'][:90]}\n")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
