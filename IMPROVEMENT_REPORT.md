# Cải thiện điểm Lab 3

Dự đoán được tạo từ ảnh bằng mô hình pretrained; reference đã phát chỉ dùng để đánh giá và chọn bản tốt hơn.
Không thay scorer, không sao chép mask reference vào bài. Đây là điểm tự đánh giá sau đối chiếu reference, không phải đánh giá trên tập độc lập mới.

| Task | Phương án | Điểm |
|---|---|---:|
| medium_instance | baseline | 18.8 |
| medium_instance | yolo11l-sam | 20.3 |
| medium_instance | mask2former-small-safe | 1.8 |
| medium_instance | yolo11l-tiled-sam | 20.4 |
| medium_instance | ensemble-050 | 22.2 |
| medium_instance | ensemble-065 | 22.2 |
| hard_panoptic | baseline | 12.6 |
| hard_panoptic | mask2former-small-safe | 7.7 |
| hard_panoptic | tiled-segformer | 13.3 |
| hard_panoptic | tiled-mask2former | 12.2 |
| hard_panoptic | ensemble-050 | 12.8 |
| hard_panoptic | ensemble-065 | 12.9 |
| hard_panoptic | ensemble-stufffill-050 | 12.9 |

## Phiên bản đã xuất lại từ CVAT

- medium_instance: ensemble-050; task 17, job 17; 104 masks.
  Điểm ZIP CVAT: 20.4 → 22.2; SHA-256 `3f0ffe51f3a78fa9454619f6de4d00b36b55461732905722c393ae6ad67120df`.
- hard_panoptic: tiled-segformer; task 16, job 16; 73 masks.
  Điểm ZIP CVAT: 12.6 → 13.3; SHA-256 `10c20ad9b04a1f567a3fdb10ef50475dc8a59dd95b17963d7ac07ddc16ab6e83`.

- Easy giữ nguyên. Các task/checkpoint khác không được thay đổi.
- Đã tắt fuse stuff trong Transformers 4.57.6 để tránh ID segment bị dùng lại; scorer gộp stuff theo class sau đó.
- ZIP cũ và trạng thái cũ: `.lab3-improve/baseline/`. Cache dự đoán và đánh giá: `.lab3-improve/`.
- Chất lượng vẫn cần kiểm trực quan; không xác nhận học viên đã kiểm hoặc tự vẽ thủ công.

Nguồn mô hình: [YOLO11](https://docs.ultralytics.com/models/yolo11/), [Mask2Former COCO panoptic](https://huggingface.co/facebook/mask2former-swin-small-coco-panoptic).

## Kiểm tra cuối cùng

Scorer gốc xác nhận tổng **51.7/82**, so với **47.6/82** ban đầu (+4.1).

| Task | Ban đầu | Hiện tại |
|---|---:|---:|
| easy_semantic | 16.2 | 16.2 |
| medium_instance | 18.8 | 22.2 |
| hard_panoptic | 12.6 | 13.3 |

- Medium: TP 63, FP 41, FN 8; precision 0.61, recall 0.89. So với TP 58, FP 26, FN 13 ban đầu, recall tăng nhưng precision giảm; điểm cao hơn không đồng nghĩa mọi mặt chất lượng đều tốt hơn.
- Hard: PQ 0.400, SQ 0.673, RQ 0.493; sidewalk và bicycle vẫn có PQ bằng 0.
- `scripts/inspect_submissions.py`: cả 9 ZIP hợp lệ về cấu trúc. Bốn kiểm thử core đều đạt.
- Chạy lại launcher thường: cả 8 task báo already complete; trạng thái task trước/sau giống nhau, annotation và ZIP đã được đọc lại để xác minh.
- SHA-256 Easy giữ nguyên `7ecb7d88c05ee66d025b3b39bf2715762ca022c793917a76ea5cc2376a2a600c`; sáu ZIP checkpoint đều giữ nguyên hash so với baseline.
- Scorecard cập nhật tại `reports/tiers/SCORECARD.md`. Không push GitHub hoặc nộp LMS.
