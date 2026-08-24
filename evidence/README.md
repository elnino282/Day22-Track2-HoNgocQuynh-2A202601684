# Evidence — Day 22 LangSmith + Prompt Versioning

LangSmith project cấu hình: `day22-ho-ngoc-quynh`.

## Danh sách bắt buộc

| Tệp | Nội dung cần nhìn thấy |
|---|---|
| `01_langsmith_traces.png` | LangSmith Runs của bước 1, tối thiểu 50 root traces `rag-query` |
| `02_prompt_hub.png` | Hai prompt `ho-ngoc-quynh-rag-prompt-v1` và `ho-ngoc-quynh-rag-prompt-v2` |
| `02_ab_routing_log.txt` | 50 dòng request có cả nhãn `prompt-v1` và `prompt-v2` |
| `03_ragas_scores.png` | Bảng bốn metric V1/V2 và thông báo faithfulness target |
| `03_ragas_report.json` | Report thật được bước 3 tự động sao chép từ `data/` |
| `04_pii_demo_log.txt` | 6 case; email, phone, SSN và credit card đều bị redact |
| `04_json_demo_log.txt` | 5 case; fences, single quotes, trailing comma và fallback |

## Phân tích V1 và V2

V1 ưu tiên câu trả lời trực tiếp trong 1–3 câu, nên thường giảm số claim và có
lợi thế về faithfulness. V2 bắt buộc tách `Answer`, `Evidence`, `Confidence`,
nên dễ kiểm tra nguồn hơn nhưng có thể phát sinh thêm claim khi diễn đạt. Cả hai
prompt đều cấm kiến thức ngoài context. Báo cáo bước 3 lưu `winner_by_mean` và
phần `analysis` dựa trên điểm thật; chỉ kết luận phiên bản thắng sau khi report
được tạo thành công.

## Kiểm tra nhanh

```powershell
Get-ChildItem .\evidence | Select-Object Name, Length
Get-Content -Raw .\data\ragas_report.json | ConvertFrom-Json
Select-String -Path .\evidence\02_ab_routing_log.txt -Pattern 'prompt-v1','prompt-v2'
```

Không đưa `.env`, API key hoặc ảnh/report giả vào thư mục này.
