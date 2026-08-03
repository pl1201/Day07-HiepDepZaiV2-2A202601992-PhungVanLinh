# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phùng Văn Linh - 2A202601992

**Nhóm:** Hiệp Đẹp Zai V2

**Ngày:** 03/08/2026

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

## 1. Khởi động (5 điểm)

### Độ tương tự cosine

Cosine cao nghĩa là hai vector gần cùng hướng, tức hai văn bản có cách biểu diễn ngữ nghĩa gần nhau dù độ dài có thể khác nhau.

- Cặp cao: “Sinh viên đăng ký học phần trên cổng học vụ.” / “Người học ghi danh môn học qua hệ thống đào tạo.” Hai câu khác từ nhưng cùng ý định đăng ký môn.
- Cặp thấp: “Thư viện cho phép mượn sách.” / “Hôm nay trời mưa rất lớn.” Hai câu thuộc hai chủ đề không liên quan.

Cosine ưu tiên hướng thay vì độ lớn vector, nên ít bị ảnh hưởng bởi độ dài văn bản. Euclidean đo khoảng cách tuyệt đối và có thể xem hai vector cùng hướng nhưng khác độ lớn là xa nhau.

### Bài toán chunking

Với 10.000 ký tự, `chunk_size=500`, `overlap=50`:

`ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23 chunks`.

Khi overlap tăng lên 100: `ceil((10000 - 100) / 400) = 25 chunks`. Chồng lấn lớn hơn làm tăng số chunk và chi phí lưu/tìm kiếm, đổi lại thông tin nằm sát biên ít bị mất ngữ cảnh.

## 2. Hướng tiếp cận của tôi (10 điểm)

### Chunking

`SentenceChunker` dùng regex `(?<=[.!?])\s+`, giữ dấu câu trong câu trước rồi gom tối đa N câu. Chuỗi rỗng chỉ trả danh sách rỗng; N luôn tối thiểu là 1.

`RecursiveChunker` thử separator theo thứ tự ưu tiên, gom các mảnh nhỏ đến gần `chunk_size`, rồi đệ quy với separator tiếp theo cho phần quá dài. Base case là đoạn đã đủ ngắn; nếu hết separator thì cắt cứng để thuật toán luôn kết thúc.

Ý tưởng cá nhân là `PolicySectionChunker`: nhận diện Markdown heading và các nhãn `Chương/Phần/Mục/Điều`, coi chúng là “neo ngữ nghĩa”. Nếu một mục quá dài, nội dung được chia tiếp nhưng tiêu đề được lặp lại ở từng chunk con. Nhờ vậy một đoạn về “Điều 5 — Điều kiện đăng ký” vẫn tự giải thích được khi đứng riêng, thay vì trở thành một mẩu văn bản mất chủ đề.

### EmbeddingStore

Mỗi record chứa ID duy nhất, nội dung, bản sao metadata và embedding; `doc_id` được bổ sung nếu đầu vào chưa có. Search nhúng query một lần, tính dot product với các record, sắp xếp giảm dần và trả cấu trúc không làm lộ embedding nội bộ.

`search_with_filter` lọc metadata **trước** khi tính similarity, vừa đúng ngữ nghĩa vừa giảm số phép tính. `delete_document` tìm toàn bộ chunk có cùng `metadata.doc_id`, xóa đồng bộ ở bộ nhớ và ChromaDB (nếu backend này có mặt).

### KnowledgeBaseAgent

Agent lấy top-k chunk, đánh số nguồn `[1]`, `[2]`, đưa cả URL/source và nội dung vào prompt. Prompt buộc câu trả lời chỉ dựa trên ngữ cảnh, phải nói “chưa đủ thông tin” khi thiếu dữ liệu và không được tự suy đoán quy định.

## 3. Hoàn thiện code (30 điểm)

```text
..........................................                               [100%]
42 passed in 0.07s
```

**Số test vượt qua: 42 / 42.** Demo `main.py` cũng chạy trọn pipeline và nạp được 3 chunk. Đã thêm xử lý UTF-8 để demo tiếng Việt không lỗi trên Windows CP1252.

## 4. Dự đoán độ tương tự (5 điểm)

Trước khi chạy, tôi dự đoán theo ngữ nghĩa tự nhiên; điểm thực tế được đo bằng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, cùng backend với benchmark nhóm.

| # | Câu A (rút gọn) | Câu B (rút gọn) | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Đăng ký học phần trên cổng học vụ | Ghi danh qua hệ thống đào tạo | Cao | 0.7929 | Có |
| 2 | Thư viện cho mượn sách | Mượn tài liệu tại thư viện | Cao | 0.7122 | Có |
| 3 | Hết hạn điều chỉnh lớp | Trời mưa lớn | Thấp | 0.8704 | Không |
| 4 | Yêu cầu môn tiên quyết | Hoàn thành môn nền tảng trước | Cao | 0.7638 | Có |
| 5 | Mang thẻ khi mượn tài liệu | Python là ngôn ngữ lập trình | Thấp | 0.1410 | Có |

Bất ngờ nhất là cặp 3 đạt 0.8704 dù hai câu khác chủ đề. Điều này cho thấy một điểm cosine riêng lẻ không đủ để khẳng định hai câu đồng nghĩa: model có thể bị ảnh hưởng bởi cấu trúc câu ngắn hoặc vùng biểu diễn dày của dữ liệu tiếng Việt. Kết quả cần được đọc tương đối trong một tập ứng viên và kiểm tra bằng benchmark có nhãn, thay vì áp một ngưỡng tuyệt đối cho mọi câu.

## 5. Kết quả truy xuất cá nhân (10 điểm)

Tôi chạy đúng corpus và 5 câu hỏi chung của nhóm bằng `PolicySectionChunker(chunk_size=350)`, multilingual MiniLM và `top_k=3`. Query 5 dùng `metadata_filter={"audience": "student"}` theo yêu cầu K3. Agent được đánh giá bằng cách đối chiếu thủ công context truy xuất với gold answer trong corpus; nội dung dưới đây không bổ sung dữ kiện ngoài nguồn.

| # | Query | Top-1 | Score | Liên quan? | Nhận xét grounded answer |
|---|---|---|---:|---|---|
| 1 | Sau khi đăng ký học phần, sinh viên kiểm tra tổng học phí ở đâu? | Học phí: kiểm tra sau thanh toán (`ueh-tuition-payment`) | 0.733336 | Top-1 không; chunk đúng hạng 2 | Chọn “In phiếu đóng tiền” để kiểm tra danh sách học phần và tổng học phí. |
| 2 | Sau khi đóng học phí, cần đối chiếu những thông tin nào trên Portal? | Học phí: kiểm tra sau thanh toán (`ueh-tuition-payment`) | 0.543298 | Có | Đối chiếu học phí đã nộp và thời khóa biểu đã cập nhật. |
| 3 | Một cuốn sách thư viện UEH được mượn và gia hạn trong bao lâu? | Thư viện: mượn/gia hạn (`ueh-library-borrowing`) | 0.800108 | Có | Mượn 20 ngày; được gia hạn một lần thêm 20 ngày. |
| 4 | Những sinh viên nào được ưu tiên khi đăng ký ký túc xá UEH? | Ký túc xá: nhóm ưu tiên (`ueh-dormitory-registration`) | 0.742934 | Có | Hộ nghèo/cận nghèo và người có thành tích nổi bật kèm minh chứng. |
| 5 | Học bổng hỗ trợ học tập mở đăng ký mấy lần mỗi năm? | Học bổng hỗ trợ học tập (`ueh-scholarship-policy`) | 0.777889 | Có | Hai lần mỗi năm, vào học kỳ đầu và học kỳ cuối. |

**Top-3 chứa chunk liên quan: 5/5; Top-1 đúng: 4/5; Recall@3 = 100%; MRR = 0.90.** Failure case là Query 1: cụm “tổng học phí” kéo tài liệu học phí lên hạng 1, trong khi ý định thật là bước cuối của quy trình đăng ký học phần và chunk đúng đứng hạng 2. `PolicySectionChunker` vẫn giữ tiêu đề cùng nội dung nên chunk tự giải thích tốt, nhưng section coherence không đảm bảo thứ hạng cao nhất. Cải tiến phù hợp là thêm metadata `workflow=course-registration`, category boost hoặc hybrid retrieval; chỉ tăng `top_k` sẽ thêm nhiễu mà không xử lý nguyên nhân.

Điều hay nhất tôi học được từ phần so sánh của nhóm là chất lượng chunk và chất lượng xếp hạng không hoàn toàn giống nhau. Fixed-size đạt MRR 1.00 trên corpus ngắn, còn section-aware dễ đọc và truy vết hơn nhưng MRR 0.90; vì vậy cần chọn chiến lược bằng dữ liệu benchmark đúng dạng tài liệu thực tế, không chỉ bằng cảm giác chunk “đẹp”.

### Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Core implementation | 30 / 30 |
| Dự đoán similarity | 5 / 5 |
| Kết quả truy xuất cá nhân | 9 / 10 |
| **Tổng** | **59 / 60** |

Query 1 được tự chấm 1/2 vì chunk đúng chỉ ở hạng 2; bốn query còn lại đạt 2/2. Kết quả có thể tái lập bằng `python scripts/benchmark_group.py --provider local`; đọc mục `policy_section_350` trong JSON đầu ra.
