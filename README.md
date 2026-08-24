# Chào mừng các bạn đến với Day 22: LangSmith + Prompt Versioning

## Tổng quan

Trong lab này, bạn sẽ xây dựng một hệ thống hỏi đáp hoàn chỉnh tích hợp nhiều công nghệ AI hiện đại:

- **RAG Pipeline**: Xây dựng pipeline Retrieval-Augmented Generation sử dụng FAISS làm vector store và LangChain để kết nối các thành phần.
- **LangSmith Tracing**: Theo dõi và quan sát toàn bộ luồng xử lý của ứng dụng LLM thông qua LangSmith dashboard.
- **Prompt Hub & A/B Testing**: Quản lý phiên bản prompt trên LangSmith Prompt Hub và thực hiện A/B routing để so sánh hiệu quả giữa các phiên bản.
- **RAGAS Evaluation**: Đánh giá chất lượng hệ thống RAG theo 4 chỉ số định lượng: faithfulness, answer relevancy, context recall, context precision.
- **Guardrails AI**: Triển khai các bộ kiểm duyệt tự động để phát hiện thông tin cá nhân (PII) và sửa lỗi định dạng JSON trong đầu ra của LLM.

---

## Mục tiêu học tập

Sau khi hoàn thành lab này, bạn sẽ có thể:

- Xây dựng và triển khai RAG pipeline hoàn chỉnh với LangChain LCEL và FAISS vector store.
- Tích hợp LangSmith để theo dõi, gỡ lỗi và phân tích hiệu suất của ứng dụng LLM trong thực tế.
- Quản lý vòng đời prompt bằng LangSmith Prompt Hub và thực hiện A/B testing có kiểm soát.
- Đánh giá hệ thống RAG một cách định lượng bằng framework RAGAS với các chỉ số chuẩn công nghiệp.
- Áp dụng Guardrails AI để xây dựng validator tùy chỉnh nhằm bảo vệ đầu ra của LLM khỏi dữ liệu nhạy cảm và lỗi định dạng.

---

## Yêu cầu trước

Trước khi bắt đầu, hãy đảm bảo bạn đã có:

- **Python 3.10 trở lên** — kiểm tra bằng lệnh `python --version`
- **API key** của ít nhất một trong các nhà cung cấp LLM sau:
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Anthropic Claude (`ANTHROPIC_API_KEY`)
  - OpenRouter (`OPENROUTER_API_KEY`)
  - Ollama (chạy local, không cần API key)
- **Tài khoản LangSmith** — đăng ký miễn phí tại [smith.langchain.com](https://smith.langchain.com) và lấy API key

---

## Cài đặt môi trường

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

> Lần đầu cài có thể mất 5–10 phút do nhiều gói phụ thuộc.

### 2. Cấu hình tệp `.env`

Sao chép tệp mẫu và điền thông tin của bạn:

```bash
cp .env.example .env
```

Mở tệp `.env` và điền các giá trị sau:

```env
# LangSmith — bắt buộc cho tất cả các bước
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=day22-lab
LANGCHAIN_TRACING_V2=true

# Chọn một trong các provider bên dưới
PROVIDER=openai

# OpenAI (nếu dùng PROVIDER=openai)
OPENAI_API_KEY=sk-...

# Google Gemini (nếu dùng PROVIDER=gemini)
GOOGLE_API_KEY=AIza...

# Anthropic (nếu dùng PROVIDER=anthropic)
ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter (nếu dùng PROVIDER=openrouter)
OPENROUTER_API_KEY=sk-or-...
```

### 3. Chọn LLM provider

Đặt biến `PROVIDER` trong `.env` thành một trong các giá trị sau:

| Giá trị      | Nhà cung cấp      | Ghi chú                         |
|--------------|-------------------|---------------------------------|
| `openai`     | OpenAI GPT        | Mặc định, ổn định nhất          |
| `gemini`     | Google Gemini     | Miễn phí với quota giới hạn     |
| `anthropic`  | Anthropic Claude  | Chất lượng cao                  |
| `ollama`     | Ollama (local)    | Không cần API key, cần GPU/CPU  |
| `openrouter` | OpenRouter        | Tổng hợp nhiều model            |

### 4. Xác minh cài đặt

```bash
cd src && python config.py
```

Nếu không có lỗi xuất hiện, bạn đã sẵn sàng bắt đầu.

---

## Cấu trúc dự án

```
Lab/
├── src/
│   ├── config.py                      # Tải .env, cấu hình providers
│   ├── utils/
│   │   ├── llm_factory.py             # Factory tạo LLM và Embeddings (5 providers)
│   │   └── data_loader.py             # Load knowledge base, chunk, build FAISS
│   ├── qa_pairs.py                    # 50 cặp câu hỏi + đáp án chuẩn
│   ├── 01_langsmith_rag_pipeline.py   # Bước 1: RAG + LangSmith tracing
│   ├── 02_prompt_hub_ab_routing.py    # Bước 2: Prompt Hub + A/B routing
│   ├── 03_ragas_evaluation.py         # Bước 3: RAGAS evaluation (~15-30 phút)
│   ├── 04_guardrails_validator.py     # Bước 4: Guardrails AI validators
│   └── run_all.py                     # Chạy tất cả các bước
├── data/
│   ├── knowledge_base.txt             # Tài liệu nguồn cho RAG
│   └── ragas_report.json              # Được tạo ra ở Bước 3
├── evidence/                          # Nộp thư mục này lên GitHub
│   ├── 01_langsmith_traces.png
│   ├── 02_prompt_hub.png
│   ├── 02_ab_routing_log.txt
│   ├── 03_ragas_scores.png
│   ├── 03_ragas_report.json
│   ├── 04_pii_demo_log.txt
│   └── 04_json_demo_log.txt
├── .env.example                        # Template biến môi trường
├── requirements.txt
├── README.md
├── rubric.md
└── Guide.md
```

---

## Các nhiệm vụ

Lab được chia thành 4 nhiệm vụ, mỗi nhiệm vụ 25 điểm (tổng 100 điểm):

| Nhiệm vụ | Tên                              | Điểm | Thời gian ước tính   |
|----------|----------------------------------|------|----------------------|
| 1        | RAG Pipeline với LangSmith       | 25đ  | 25–45 phút           |
| 2        | Prompt Hub & A/B Routing         | 25đ  | 20–30 phút           |
| 3        | RAGAS Evaluation                 | 25đ  | 45–75 phút           |
| 4        | Guardrails AI Validators         | 25đ  | 20–30 phút           |

**Nhiệm vụ 1 — RAG Pipeline với LangSmith (25đ):** Xây dựng vector store từ knowledge base, tạo RAG chain, và tích hợp `@traceable` để ghi lại ít nhất 50 traces trên LangSmith dashboard.

**Nhiệm vụ 2 — Prompt Hub & A/B Routing (25đ):** Soạn 2 system prompt có ngữ nghĩa khác biệt, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi theo hash của `request_id`.

**Nhiệm vụ 3 — RAGAS Evaluation (25đ):** Chạy 50 cặp QA qua cả 2 phiên bản prompt, xây dựng `EvaluationDataset`, tính 4 chỉ số RAGAS, và đạt faithfulness ≥ 0.8 với ít nhất 1 phiên bản.

**Nhiệm vụ 4 — Guardrails AI Validators (25đ):** Triển khai `PIIDetector` tự động che thông tin cá nhân và `JSONFormatter` tự động sửa JSON lỗi từ đầu ra của LLM.

---

## Chạy lab

### Trạng thái triển khai

Mã nguồn đã hoàn thiện toàn bộ các phần chấm tự động: RAG/FAISS/LCEL,
LangSmith tracing, push + pull Prompt Hub, MD5 A/B routing, dataset và bốn
metric RAGAS, hai custom Guardrails validators, fallback JSON, lưu report và
`run_all.py`. Chạy regression test offline trước khi gọi API:

```powershell
& .\venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Kết quả đã xác minh

| Hạng mục | Kết quả |
|---|---|
| Regression tests | 11/11 tests pass offline |
| LangSmith Step 1 | 54 root traces `rag-query`; 53 có đủ input + output |
| LangSmith Step 2 | 50 root traces `ab-rag-query`; không có lỗi |
| Prompt Hub | Cả V1 và V2 đều tồn tại và pull được từ Hub |
| A/B routing | 50 câu: V1 = 19, V2 = 31; routing theo MD5 tất định |
| RAGAS V1 | faithfulness 0.9345; answer relevancy 0.9230; context recall 1.0000; context precision 0.9464 |
| RAGAS V2 | faithfulness 0.9206; answer relevancy 0.9214; context recall 1.0000; context precision 0.9545 |
| Guardrails | 6 PII cases và 5 JSON cases chạy thành công |
| Bảo mật | `.env` bị Git ignore; không commit API key |

Báo cáo RAGAS chính thức nằm tại `data/ragas_report.json`; bản sao nộp bài ở
`evidence/03_ragas_report.json`. Không chỉnh tay các giá trị điểm trong hai tệp
này. Kiểm tra LangSmith ở chế độ chỉ đọc bằng:

```powershell
& .\venv\Scripts\python.exe .\scripts\audit_langsmith.py
```

Ba ảnh PNG trong checklist evidence đã được bổ sung bằng ảnh chụp thật từ
LangSmith và terminal. Không thay ảnh bằng file log hoặc ảnh minh họa.

### Chạy từng bước riêng lẻ

```bash
cd src

# Bước 1: RAG Pipeline với LangSmith tracing
python 01_langsmith_rag_pipeline.py

# Bước 2: Prompt Hub và A/B routing
python 02_prompt_hub_ab_routing.py

# Bước 3: RAGAS evaluation (mất 15–30 phút)
python 03_ragas_evaluation.py

# Bước 4: Guardrails AI validators
python 04_guardrails_validator.py
```

### Chạy toàn bộ lab

```bash
cd src && python run_all.py
```

### Chạy một bước cụ thể

```bash
cd src && python run_all.py --step 3
```

### Lệnh PowerShell tạo evidence cho bước 2–4

PowerShell 5.1 thường dùng code page 437/cp1252 và có thể làm vỡ dấu tiếng Việt.
Launcher dưới đây tự chuyển console, Python và các log sang UTF-8. Chạy từ thư
mục gốc repository:

```powershell
# Bước 2: tự lưu evidence/02_ab_routing_log.txt dạng UTF-8
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Step 2

# Bước 3: 100 RAG answers + 4 RAGAS metrics cho mỗi version
# Script tự tạo data/ragas_report.json và evidence/03_ragas_report.json
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Step 3

# Bước 4: tạo hai log riêng đúng rubric
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Step 4
```

Task 3 dùng `gpt-4o` theo `OPENAI_MODEL`. Với OpenAI Responses-compatible
proxy, `answer_relevancy` chạy `strictness=1` để tránh request `n=3` không được
hỗ trợ. Sau khi tạo đủ 100 RAG outputs, script lưu restart cache tại
`data/ragas_inputs_cache.json`; cache tự vô hiệu nếu model, prompt, QA hoặc
knowledge base thay đổi.

Khi cần đọc file trực tiếp trong Windows PowerShell 5.1, luôn chỉ rõ encoding:

```powershell
Get-Content -Raw -Encoding UTF8 .\Guide.md
Get-Content -Raw -Encoding UTF8 .\rubric.md
```

Sau bước 2, chụp Prompt Hub thành `evidence/02_prompt_hub.png`. Sau bước 3,
chụp bảng so sánh cuối terminal thành `evidence/03_ragas_scores.png`. Không
dùng ảnh minh họa hoặc report tự điền; evidence phải đến từ lần chạy API thật.

---

## Nộp bài

Repository public:
[github.com/elnino282/Day22-Track2-HoNgocQuynh-2A202601684](https://github.com/elnino282/Day22-Track2-HoNgocQuynh-2A202601684)

LangSmith project: `day22-ho-ngoc-quynh`. Trước khi nộp, bật chế độ chia sẻ
công khai của project và gửi **URL share công khai**, không gửi URL dashboard
chỉ hoạt động trong phiên đăng nhập cá nhân.

### 1. Tạo GitHub repository

Tạo repository public mới trên GitHub với tên ví dụ `day22-langsmith-lab`.

### 2. Thu thập bằng chứng (evidence)

Đảm bảo thư mục `evidence/` chứa đầy đủ 7 tệp sau:

```
evidence/
├── 01_langsmith_traces.png      ← Ảnh chụp màn hình LangSmith dashboard (≥ 50 traces)
├── 02_prompt_hub.png            ← Ảnh chụp màn hình Prompt Hub (2 phiên bản)
├── 02_ab_routing_log.txt        ← Output console của bước 2
├── 03_ragas_scores.png          ← Ảnh chụp terminal hiển thị điểm RAGAS
├── 03_ragas_report.json         ← Báo cáo JSON từ RAGAS
├── 04_pii_demo_log.txt          ← Output console của PII detector
└── 04_json_demo_log.txt         ← Output console của JSON formatter
```

Ba tệp PNG đã có trong repository. Nếu cần chụp lại
`01_langsmith_traces.png`, nên lọc root run theo tên `rag-query`; không chỉ
chụp 100 run mới nhất vì các run RAGAS có thể che khuất trace của Bước 1.

### 3. Lưu output console vào tệp

Sử dụng lệnh `tee` để vừa in ra màn hình vừa lưu vào tệp:

```bash
python script.py | tee evidence/output.txt
```

Ví dụ cụ thể:

```bash
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
python 04_guardrails_validator.py  | tee ../evidence/04_pii_demo_log.txt
```

### 4. Push lên GitHub và nộp

```bash
git init
git add .
git commit -m "Day 22: LangSmith + Prompt Versioning lab submission"
git remote add origin https://github.com/<tên-của-bạn>/day22-langsmith-lab.git
git push -u origin main
```

Nộp URL GitHub repository và URL LangSmith project của bạn qua cổng nộp bài của khóa học.

---

## Tips và lưu ý

**LangSmith tracing — đặt biến môi trường đúng thứ tự:**
Các biến `LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`, và `LANGSMITH_PROJECT` phải được đặt **trước khi import bất kỳ thứ gì từ LangChain**. Nếu import trước khi đặt biến, tracing sẽ không hoạt động.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # Phải đặt trước
os.environ["LANGSMITH_API_KEY"]    = "..."     # Phải đặt trước
from langchain_core.prompts import ChatPromptTemplate  # Sau đó mới import
```

**RAGAS chậm — bắt đầu sớm:**
Bước 3 sẽ mất từ 15 đến 30 phút để hoàn thành do phải gọi LLM cho mỗi sample trong bộ đánh giá. Hãy bắt đầu bước này ngay khi bước 2 xong, đặc biệt nếu bạn đang dùng model có rate limit thấp.

**Guardrails AI — `on_fail` phải truyền đúng chỗ:**
Tham số `on_fail` phải được truyền vào **constructor của validator**, không phải vào `Guard.use()`:

```python
# ĐÚNG
Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

# SAI — sẽ không hoạt động đúng
Guard().use(PIIDetector(), on_fail=OnFailAction.FIX)
```

**Bảo mật — không bao giờ commit `.env`:**
Tệp `.env` chứa API key nhạy cảm. Đảm bảo `.gitignore` đã có dòng `.env` trước khi push lên GitHub. Chỉ commit tệp `.env.example` (không chứa giá trị thật). Vi phạm quy tắc này sẽ bị trừ 10 điểm tự động.

---

## Tài liệu tham khảo

| Tài liệu                    | Đường dẫn                                                          |
|-----------------------------|--------------------------------------------------------------------|
| LangSmith Docs              | https://docs.smith.langchain.com                                   |
| LangChain LCEL              | https://python.langchain.com/docs/concepts/lcel                    |
| LangSmith Prompt Hub        | https://docs.smith.langchain.com/prompt-hub                        |
| RAGAS Documentation         | https://docs.ragas.io                                              |
| Guardrails AI               | https://www.guardrailsai.com/docs                                  |
| FAISS (Facebook AI)         | https://faiss.ai                                                   |
| LangChain FAISS Integration | https://python.langchain.com/docs/integrations/vectorstores/faiss  |
