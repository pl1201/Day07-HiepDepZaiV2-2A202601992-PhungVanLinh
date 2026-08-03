# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Hiệp Đẹp Zai V2

**Thành viên:** Đinh Lê Quỳnh Phương - 2A202601865 , Đoàn Minh Hiếu - 2A202601733, Lưu Quang Nhật - 2A202601920, Nguyễn Ngọc Sơn - 2A202601948, Phùng Văn Linh - 2A202601992

**Ngày:** 03/08/2026

**Tổng điểm phần nhóm: 40** = Chất lượng bộ tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Demo (5).

## 1. Lựa chọn tài liệu (10 điểm)

### Phạm vi

Nhóm xây dựng trợ lý tra cứu năm dịch vụ thiết yếu dành cho người học UEH: đăng ký học phần, học phí, học bổng, thư viện và ký túc xá. Corpus chỉ chứa bản tóm tắt từ trang công khai chính thức, không chứa dữ liệu cá nhân hoặc nội dung sau đăng nhập.

### Danh sách tài liệu

| # | Tài liệu | Nguồn chính thức | Truy xuất / phiên bản | Ký tự | Metadata chính |
|---|---|---|---|---:|---|
| 1 | Hướng dẫn đăng ký học phần | [Cổng tư vấn UEH](https://nhaphoc.ueh.edu.vn/dinh-huong-sau-nhap-hoc/hoc-tap-tai-ueh/dao-tao-dhcq/xay-dung-ke-hoach-hoc-tap/) | 03/08/2026 / accessed-2026-08-03 | 515 | student, academic-affairs, course-registration |
| 2 | Nộp và kiểm tra học phí | [Cổng tư vấn UEH](https://nhaphoc.ueh.edu.vn/dinh-huong-sau-nhap-hoc/hoc-tap-tai-ueh/hoc-phi/) | 03/08/2026 / accessed-2026-08-03 | 465 | student, finance, tuition |
| 3 | Chính sách học bổng | [Ban Chăm sóc người học UEH](https://dsa.ueh.edu.vn/tin-tuc/chinh-sach-hoc-bong/) | 03/08/2026 / accessed-2026-08-03 | 591 | student, student-affairs, scholarship |
| 4 | Mượn, trả và gia hạn tài liệu | [Thư viện UEH](https://smartlib.ueh.edu.vn/services/academic) | 03/08/2026 / accessed-2026-08-03 | 385 | all, library, library |
| 5 | Đăng ký nội trú ký túc xá | [Ký túc xá UEH](https://kytucxa.ueh.edu.vn/student/dorm-register) | 03/08/2026 / accessed-2026-08-03 | 468 | student, dormitory, dormitory |

File `sources.csv` là manifest đối chiếu nguồn. Nội dung trong các file Markdown là diễn giải ngắn, chỉ giữ các dữ kiện cần cho gold answer và luôn gắn URL để truy vết.

### Metadata schema

| Trường | Kiểu | Ví dụ | Công dụng |
|---|---|---|---|
| `doc_id` | string | `ueh-scholarship-policy` | Định danh, xóa toàn bộ chunk của một tài liệu |
| `audience` | enum | `student`, `all` | Lọc đúng đối tượng; bắt buộc theo K3 |
| `department` | string | `student-affairs` | Giới hạn đơn vị nghiệp vụ |
| `category` | enum | `tuition`, `library` | Tách các câu có từ vựng gần nhau |
| `language` | string | `vi` | Chọn embedding/nguồn theo ngôn ngữ |
| `source_url` | URL | trang chính thức UEH | Truy vết câu trả lời |
| `retrieved_at` | ISO date | `2026-08-03` | Kiểm tra độ mới |
| `document_version` | string | `accessed-2026-08-03` | Ghim phiên bản khi trang không công bố số hiệu |
| `chunk_index` | integer | `1` | Tìm đúng đoạn trong tài liệu |

Checklist quản trị dữ liệu:

- [x] Có 5 tài liệu công khai, đúng chủ đề K3, không chứa dữ liệu cá nhân.
- [x] Mỗi tài liệu có `audience`, một trường hữu ích khác, URL, ngày truy xuất và phiên bản.
- [x] Gold answer bên dưới đều trích xuất được từ corpus, không suy đoán quy định.

## 2. Thiết kế chiến lược (15 điểm)

### Thiết lập chung

- Embedder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Cùng corpus, cùng 5 query, `top_k=3`.
- Metric chính: Recall@3; metric phân hạng bổ sung: MRR.
- Script tái lập: `python scripts/benchmark_group.py --provider local`.

### Baseline trên corpus

`ChunkingStrategyComparator` được chạy với `chunk_size=350`; bảng dưới thể hiện ba tài liệu đại diện.

| Tài liệu | Strategy | Số chunk | Độ dài TB | Mạch lạc |
|---|---|---:|---:|---|
| Đăng ký học phần | Fixed-size | 2 | 282,5 | Có overlap nhưng có thể cắt giữa câu |
| Đăng ký học phần | Sentence | 2 | 256,5 | Câu nguyên vẹn, đôi khi lẫn hai mục |
| Đăng ký học phần | Recursive | 2 | 256,5 | Tôn trọng đoạn/câu |
| Thư viện | Fixed-size | 2 | 217,5 | Đủ ngắn nhưng biên theo ký tự |
| Thư viện | Sentence | 2 | 191,5 | Câu nguyên vẹn |
| Thư viện | Recursive | 2 | 191,5 | Giữ đoạn tốt |
| Học bổng | Fixed-size | 2 | 320,5 | Chunk khá dày |
| Học bổng | Sentence | 2 | 294,5 | Giữ điều kiện nguyên vẹn |
| Học bổng | Recursive | 2 | 294,5 | Giữ ranh giới mục tương đối tốt |

### Chiến lược của năm thành viên

| Thành viên | Cấu hình | Lý do và đánh đổi |
|---|---|---|
| Đinh Lê Quỳnh Phương  - 2A202601865| `FixedSizeChunker(350, overlap=50)` | Baseline ổn định; overlap bảo vệ thông tin ở biên nhưng tạo lặp. |
| Đoàn Minh Hiếu - 2A202601733| `SentenceChunker(max_sentences_per_chunk=2)` | Giữ nguyên câu, phù hợp FAQ ngắn; số chunk tăng khi câu ngắn. |
| Lưu Quang Nhật - 2A202601920| `RecursiveChunker(chunk_size=350)` | Ưu tiên đoạn → dòng → câu → từ, cân bằng kích thước và mạch lạc. |
| Nguyễn Ngọc Sơn - 2A202601948 | `PolicySectionChunker(chunk_size=500)` | Giữ cấu trúc mục với context rộng hơn, giảm nguy cơ tách điều kiện và ngoại lệ. |
| Phùng Văn Linh - 2A202601992| `PolicySectionChunker(chunk_size=350)` | Custom theo heading/Chương/Phần/Mục/Điều; lặp tiêu đề cho chunk con để chunk đứng độc lập vẫn có chủ đề. |

Điểm sáng tạo của nhóm là `PolicySectionChunker`: thay vì coi tài liệu là chuỗi ký tự phẳng, tiêu đề được xem như “semantic address”. Cách này tạo 12 chunk, trung bình khoảng 200 ký tự, so với 10 chunk và khoảng 241–267 ký tự của các baseline. Nó phù hợp khi corpus mở rộng sang quy chế dài nhiều điều; trên corpus ngắn hiện tại, lợi ích coherence rõ hơn lợi ích xếp hạng.

### So sánh định lượng

| Thành viên | Strategy | Chunk | Recall@3 | MRR | Điểm retrieval (/10) |
|---|---|---:|---:|---:|---:|
| Đinh Lê Quỳnh Phương  - 2A202601865 | Fixed 350/50 | 10 | 100% | 1,00 | 10 |
| Đoàn Minh Hiếu - 2A202601733 | Sentence 2 | 12 | 100% | 1,00 | 10 |
| Lưu Quang Nhật - 2A202601920 | Recursive 350 | 10 | 100% | 0,90 | 9 |
| Nguyễn Ngọc Sơn - 2A202601948 | Policy section 500 | 12 | 100% | 0,90 | 9 |
| Phùng Văn Linh - 2A202601992 | Policy section 350 | 12 | 100% | 0,90 | 9 |

Fixed-size là cấu hình tốt nhất trên benchmark ngắn này: đúng tài liệu ở top-1 cho 5/5 query, điểm top-1 trung bình khoảng 0,754 và tạo ít chunk. Sentence cũng đạt MRR 1,0 và giữ câu tự nhiên hơn. Nhóm không kết luận Fixed luôn tốt nhất: khi đưa quy chế dài vào corpus, section-aware được kỳ vọng ít trộn điều kiện giữa các mục hơn và cần benchmark lại.

## 3. Câu hỏi đánh giá và chất lượng truy xuất (10 điểm)

### Năm câu hỏi chung và gold answer

| # | Query | Gold answer kiểm chứng được | Chunk nguồn |
|---|---|---|---|
| 1 | Sau khi đăng ký học phần, sinh viên kiểm tra tổng học phí ở đâu? | Chọn “In phiếu đóng tiền” để kiểm tra danh sách học phần và tổng học phí. | `ueh-course-registration`, mục “Kiểm tra và điều chỉnh” |
| 2 | Sau khi đóng học phí, cần đối chiếu những thông tin nào trên Portal? | Đối chiếu học phí đã nộp và thời khóa biểu đã cập nhật. | `ueh-tuition-payment`, mục “Kiểm tra sau thanh toán” |
| 3 | Một cuốn sách thư viện UEH được mượn và gia hạn trong bao lâu? | Mượn tiêu chuẩn 20 ngày; gia hạn một lần thêm 20 ngày. | `ueh-library-borrowing`, hai mục mượn/gia hạn |
| 4 | Những sinh viên nào được ưu tiên khi đăng ký ký túc xá UEH? | Hộ nghèo/cận nghèo và người có thành tích nổi bật kèm minh chứng thuộc các nhóm ưu tiên. | `ueh-dormitory-registration`, mục “Nhóm ưu tiên” |
| 5 | Học bổng hỗ trợ học tập mở đăng ký mấy lần mỗi năm? | Hai lần mỗi năm, vào học kỳ đầu và học kỳ cuối. | `ueh-scholarship-policy`, mục “Học bổng hỗ trợ học tập” |

Query 5 được chạy bằng `metadata_filter={"audience": "student"}` theo yêu cầu K3. Bộ lọc loại tài liệu `audience=all` khỏi tập ứng viên trước khi tính similarity; trong corpus nhỏ nó không đổi top-1, nhưng làm giảm nhiễu và bảo đảm câu trả lời lấy từ chính sách dành cho sinh viên.

### Kết quả tốt nhất (Fixed 350/50)

| # | Top-1 đúng? | Score | Top-3 có chunk đúng? | Grounded answer |
|---|---|---:|---|---|
| 1 | Có | 0,757179 | Có | Đúng gold answer, trích mục kiểm tra đăng ký |
| 2 | Có | 0,614822 | Có | Đúng hai thông tin cần đối chiếu |
| 3 | Có | 0,827360 | Có | Đúng 20 ngày + gia hạn một lần 20 ngày |
| 4 | Có | 0,779441 | Có | Đúng hai nhóm ưu tiên được nêu trong nguồn |
| 5 | Có | 0,790394 | Có | Đúng hai đợt/năm và thời điểm |

**Kết quả nhóm:** Recall@3 = 5/5; top-1 accuracy = 5/5; câu trả lời grounded được đối chiếu thủ công với gold answer và nguồn. Theo rubric: 10/10.

### Failure analysis

Failure case xuất hiện ở Query 1 với Recursive và hai Policy-section: chunk học phí được xếp top-1 (`score=0,743757` với Recursive), còn chunk đăng ký học phần đúng đứng hạng 2. Nguyên nhân là query có cụm “tổng học phí”, tạo tín hiệu mạnh cho tài liệu học phí dù ý định thật là bước cuối của quy trình đăng ký. Fixed/Sentence giữ câu đăng ký và “In phiếu đóng tiền” gần nhau nên xếp đúng.

Cải thiện đề xuất: bổ sung metadata `workflow=course-registration`, thử hybrid retrieval (semantic score + keyword/category boost), và dùng query routing để lọc `category` khi ý định rõ. Không nên chỉ tăng `top_k`, vì cách đó tăng recall nhưng làm context của agent nhiều nhiễu hơn.

## 4. Demo và bài học nhóm (5 điểm)

### Kịch bản demo

1. Chạy `pytest tests -v` để chứng minh 42/42 test pass.
2. Chạy `python scripts/benchmark_group.py --provider local`.
3. Đối chiếu Query 1 giữa Fixed và Policy-section để chỉ ra failure case “đăng ký” vs “học phí”.
4. Chạy Query 5 với/không có `audience=student`, giải thích pre-filter metadata.
5. Mở chunk trả về, chỉ `source_url`, `doc_id`, `chunk_index` và gold answer để chứng minh khả năng truy vết.

### Ba insight chính

- Backend quan trọng hơn tinh chỉnh chunk nhỏ: mock embedding cho xếp hạng gần ngẫu nhiên, trong khi multilingual embedder nâng tất cả chiến lược lên Recall@3 100%.
- Chunk coherence và retrieval rank không đồng nhất. Policy-section dễ đọc và có nguồn rõ hơn nhưng Fixed thắng MRR trên corpus rất ngắn.
- Query có nhiều ý định (“đăng ký” và “học phí”) dễ kéo sai tài liệu; metadata/hybrid retrieval giải quyết đúng nguyên nhân hơn việc chỉ tăng top-k.

Nếu làm lại, nhóm sẽ bổ sung các quy chế dài có số hiệu/ngày hiệu lực thay vì chỉ trang hướng dẫn ngắn, thêm tài liệu `audience=faculty/staff` để đo hiệu quả filter rõ hơn, và mở rộng benchmark bằng câu hỏi paraphrase khó, câu phủ định, câu hỏi nhiều bước. Nhóm cũng sẽ báo cáo Precision@3 và latency bên cạnh Recall@3/MRR.

## Tự đánh giá

| Tiêu chí | Điểm |
|---|---:|
| Chất lượng bộ tài liệu | 10 / 10 |
| Thiết kế chiến lược | 15 / 15 |
| Chất lượng truy xuất | 10 / 10 |
| Demo và bài học | 5 / 5 |
| **Tổng** | **40 / 40** |

### Khả năng tái lập

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
python scripts\benchmark_group.py --provider local
pytest tests -v
```
