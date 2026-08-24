# Lab Requirements — Day 22: LangSmith + Prompt Versioning

## Python Version
Python 3.10 or higher

## Install All Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` là nguồn phiên bản duy nhất. Các dependency chính được khóa
theo major version đã kiểm thử: LangChain 1.x, RAGAS 0.4.x và Guardrails AI
0.10–0.11.x. Việc khóa major ngăn một lần `pip install` trong tương lai âm thầm
đổi API của LCEL, RAGAS hoặc custom validators.

## Package Purposes

| Package | Used For |
|---------|---------|
| `langchain` | Core LLM framework |
| `langchain-openai` | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | FAISS vectorstore integration |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `langsmith` | LangSmith tracing, Prompt Hub client |
| `openai` | Direct OpenAI API calls |
| `faiss-cpu` | Similarity search index |
| `ragas` | RAG evaluation metrics |
| `guardrails-ai` | Output validation framework |
| `python-dotenv` | Load `.env` file |
| `tiktoken` | Token counting for text splitters |
| `datasets` | Required by RAGAS internally |
| `numpy` | Averaging RAGAS score lists |

## Important Version Notes

### RAGAS 0.4.x
- Use `from ragas.metrics import faithfulness, answer_relevancy, ...` (NOT from `ragas.metrics.collections`)
- `result[metric_name]` returns a **list** of floats for multiple samples — use `numpy.mean()` to average
- Pass `llm=` and `embeddings=` to the `evaluate()` function, not to metric constructors

### Guardrails AI 0.10.x
- `on_fail` parameter belongs in the **validator constructor**: `MyValidator(on_fail=OnFailAction.FIX)`
- `Guard.use()` accepts validator **instances**, not classes
- `Guard.validate(text)` is the main entry point

### LangChain 1.x
- Dùng `ChatOpenAI(api_key=..., base_url=..., model=...)` cho custom endpoint.
- Dùng `OpenAIEmbeddings(api_key=..., base_url=..., model=...)` cho embedding endpoint.
- RAGAS 0.4.3 còn import một VertexAI module tùy chọn đã bị gỡ khỏi
  `langchain-community` 0.4; `src/utils/ragas_compat.py` cung cấp marker shim chỉ
  cho type-check này. Lab không gọi VertexAI và các provider chính không bị đổi.

## Environment Variables

Copy this to your `.env` file:


> ⚠️ **Never commit `.env` to git.** Add it to `.gitignore`.

## Verify Installation

Run the config check:
```bash
python config.py
```

Expected output:
```
✅ Config loaded successfully
   LangSmith project : your-project-name
   OpenAI endpoint   : https://...
   Default LLM model : gpt-5.4-mini
   Embedding model   : text-embedding-3-small
```
