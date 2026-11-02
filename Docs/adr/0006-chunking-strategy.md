# ADR-0006: Chunking — heading-aware + recursive, 512/64

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Chất lượng retrieval phụ thuộc mạnh vào cách chia chunk. Chunk quá lớn → vector "loãng", tốn context; quá nhỏ → mất ngữ cảnh. Input MVP là TXT/MD, nơi heading Markdown là ranh giới ngữ nghĩa tự nhiên. BGE-M3 nhận tới 8192 token nên không bị giới hạn bởi model.

## Quyết định
1. **Markdown:** tách trước theo heading (`#`..`######`), ghi `heading_path` cho mỗi section. TXT: cả file là một section.
2. Mỗi section lớn hơn `chunk_size` → **recursive split** theo thứ tự separator: `\n\n` → `\n` → câu (`. ` `? ` `! `) → khoảng trắng; không cắt giữa từ.
3. Mặc định `chunk_size = 512` token (ước lượng), `chunk_overlap = 64` (~12%). MVP ước lượng token bằng số ký tự/4 (cấu hình được); nếu `rag eval` cho thấy lệch nhiều với tiếng Việt thì chuyển sang tokenizer thật.
4. Section quá nhỏ (< 50 token) được gộp với section kế tiếp cùng cha.
5. Khi embed, prepend heading_path vào text chunk (vd. `"Nghỉ phép > Nghỉ phép năm\n\n<nội dung>"`) để vector mang ngữ cảnh tiêu đề; lưu `content` gốc riêng.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Heading-aware + recursive (chọn) | Tôn trọng cấu trúc tài liệu, đơn giản, xác định | Cần tinh chỉnh tham số |
| Fixed-size theo ký tự | Đơn giản nhất | Cắt ngang ý, ngang câu |
| Semantic chunking (theo độ tương đồng câu) | Ranh giới ngữ nghĩa tốt | Tốn embedding lúc ingest, khó kiểm chứng |
| Một chunk mỗi file | Không mất ngữ cảnh | Vector loãng, vượt context LLM |

## Hệ quả
- Kết quả search hiển thị được `source › heading`, giúp người dùng kiểm chứng ngay.
- Tham số sẽ được tinh chỉnh dựa trên số đo ở M5 và ghi baseline vào [06-evaluation-and-testing.md](../06-evaluation-and-testing.md).

## Addendum 2026-11-02 (làm rõ khi triển khai M1–M2, không đổi quyết định)
- **TXT:** thay vì "cả file là một section", mỗi **đoạn văn** (ngăn bởi dòng trống) là một section; đoạn < `min_section_tokens` vẫn được gộp vào đoạn sau. Lý do đo được: `samples/office-notes.txt` (~1.000 ký tự, 5 chủ đề) thành 1 chunk "loãng", câu hỏi về gửi ô tô rơi xuống rank 4; sau khi sửa lên rank 1 (score 0.448 → 0.690).
- **Gộp section nhỏ:** section gộp mang `heading_path` của section phía sau (thường là heading con, đã chứa heading cha), nội dung giữ nguyên các dòng heading.
