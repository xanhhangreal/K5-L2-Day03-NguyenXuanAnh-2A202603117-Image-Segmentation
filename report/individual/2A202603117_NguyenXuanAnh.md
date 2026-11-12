# Báo cáo thực hành Day 3 — Image Segmentation

- Học viên: Nguyễn Xuân Anh — mã học viên: **2A202603117**.
- Ngày thực hiện: 11–12/11/2026; CVAT local tại `http://localhost:8080`, tài khoản `anhnguyenkaz`.
- Công cụ: CVAT, WSL Ubuntu, Docker, Python; YOLO11n/YOLO11l, SAM ViT-B CPU, SegFormer-B0 Cityscapes; thử nghiệm Mask2Former COCO panoptic; trợ lý Codex hỗ trợ triển khai, chạy tool và tổng hợp báo cáo.
- Phạm vi đóng góp: Easy được học viên thông báo đã hoàn thành trước khi chạy tự động. Tám task còn lại do pipeline tự động tạo mask, lưu vào CVAT và export. Những quan sát kỹ thuật dưới đây do trợ lý kiểm tra; không nhận là học viên đã tự vẽ hoặc tự kiểm toàn bộ ảnh.

## 1. Bài đã nộp

Các ZIP đã được chuẩn bị để đưa lên fork. “Hoàn thành” trong bảng là đã có annotation lưu trên CVAT và ZIP xuất được, không phải xác nhận mọi mask đều đúng hay đã đáp ứng đủ bước thực hành thủ công.

| Task | File ZIP đúng tên | Số ảnh đã lưu/export | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | `submissions/easy_semantic.zip` | 3 / 3 | 20 |
| medium_instance | `submissions/medium_instance.zip` | 3 / 3 | 32 |
| hard_panoptic | `submissions/hard_panoptic.zip` | 2 / 2 | 30 |
| cp1_holes | `submissions/cp1_holes.zip` | 1 / 1 | 3 |
| cp2_slice | `submissions/cp2_slice.zip` | 1 / 1 | 3 |
| cp5_occlusion | `submissions/cp5_occlusion.zip` | 1 / 1 | 3 |
| cp3_thin | `submissions/cp3_thin.zip` | 1 / 1 | 3 |
| cp4_curb | `submissions/cp4_curb.zip` | 1 / 1 | 3 |
| cp6_coverage | `submissions/cp6_coverage.zip` | 1 / 1 | 3 |
| **Tổng tối đa** | **9 ZIP** | **14 / 14** | **100** |

Semantic dùng **Segmentation mask 1.1**; instance/panoptic dùng **COCO 1.0**. Nội dung ZIP được giữ nguyên từ CVAT. Không có task export lỗi ở lần kiểm cuối. Cả 9 ZIP đạt kiểm tra cấu trúc bằng `scripts/inspect_submissions.py`; kết quả này không xác nhận biên, số vật hay lớp đều đúng.

## 2. Một quyết định trước khi dùng gợi ý

**Chưa có object Medium tự vẽ trước khi dùng gợi ý được ghi nhận.** Các mask Medium do tool tạo tự động; không dùng chúng để thay bằng chứng tự vẽ theo README. Đây là phần yêu cầu thực hành còn thiếu.

Quyết định kỹ thuật thực tế của pipeline, ghi để phân biệt với thao tác tự vẽ:

- Ảnh `000000181542.jpg`, nhóm người và xe máy trong cảnh đường phố: phát hiện riêng các lớp `person`, `motorcycle`, `car`, `bus`, `bicycle` thuộc taxonomy của task rồi dùng box làm prompt cho SAM. Bản cuối có 36 mask trên ảnh này; đây là số dự đoán, không phải số vật đúng đã xác nhận.
- Giữ định danh từng vật; không gộp hai vật chỉ vì cùng class. Các mảng rời của một mask được giữ trong RLE; không chủ động vẽ thêm phần bị che.
- Sau khi thử các phương án, chọn bản ensemble cho Medium theo kết quả scorer trên reference đã phát. Việc chọn dựa vào metric không thay thế kiểm thủ công từng object. Bản cuối có thêm vật được ghép đúng nhưng cũng có nhiều vật thừa hơn.
- Chưa khẳng định mask cụ thể nào được học viên sửa bằng brush hoặc kiểm biên thủ công.

## 3. Một lỗi tìm thấy và hành động cải thiện

- **Task/vùng:** Medium, nhóm vật nhỏ và bị che trong ba ảnh `000000181542.jpg`, `000000373353.jpg`, `000000458325.jpg`.
- **Loại lỗi:** thiếu vật và thừa vật. Scorer ban đầu ghi TP 58, FP 26, FN 13. Đây là bằng chứng tổng hợp của cả task, không suy ra vị trí từng FN từ con số đó.
- **Hành động đã thực hiện:** dùng YOLO11l trên toàn ảnh và các crop để phát hiện vật nhỏ; đưa box sang SAM; kết hợp dự đoán theo class và độ chồng mask, loại trùng. Dự đoán tạo từ ảnh, không chép mask ground truth.
- **Lưu/export:** đã lưu 104 mask ở CVAT task/job **17/17** và export lại `medium_instance.zip` bằng COCO 1.0; đọc lại annotation và chấm chính ZIP xuất từ CVAT.
- **Kết quả:** TP 63, FP 41, FN 8; recall 0.82 → 0.89, precision 0.69 → 0.61. Bỏ sót giảm nhưng dự đoán thừa tăng; lỗi chưa được xử lý triệt để. Không nhận là đã sửa từng lỗi bằng tay.
- **Hard:** bản YOLO11l chạy crop + SAM kết hợp SegFormer được lưu tại task/job **16/16**, 73 mask; PQ tăng từ 0.390 lên 0.400. Sidewalk và bicycle vẫn có PQ bằng 0.

Kết quả tự đánh giá bằng scorer gốc sau khi có reference:

| Task | Metric ban đầu → bản cuối | Điểm tự đánh giá ban đầu → bản cuối |
| --- | --- | --- |
| Easy | mIoU 0.765 → 0.765 | 16.2 → 16.2 / 20 |
| Medium | mean matched IoU × recall 0.664 → 0.712 | 18.8 → 22.2 / 32 |
| Hard | PQ 0.390 → 0.400 | 12.6 → 13.3 / 30 |
| **Ba tier** | | **47.6 → 51.7 / 82** |

Đây là phản hồi tự đánh giá, không phải điểm chính thức /100. Không tự chấm sáu checkpoint, PASS hoặc bonus. Reference được dùng để đánh giá và chọn phiên bản nên đây không phải kết quả trên tập kiểm tra độc lập. Easy và sáu ZIP checkpoint được giữ nguyên trong lần cải thiện.

Chi tiết: [scorecard](../../reports/tiers/SCORECARD.md), [báo cáo cải thiện](../../IMPROVEMENT_REPORT.md), [nhật ký CVAT và cấu hình](../../AUTOMATION_RUN_REPORT.md). Chạy lại tool đã xác minh task/annotation/ZIP không bị nhân đôi hoặc thay đổi.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Các ca sau dựa trên quan sát overlay của trợ lý, không ghi thành trải nghiệm kiểm thủ công của học viên.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| `cp2_slice/000000017627.jpg`, xe lớn bị cắt ở mép trái | Bỏ qua vì vật không nằm trọn ảnh / gán phần nhìn thấy nếu thuộc class của task | Overlay chưa thấy mask cho phần xe lớn này; quy tắc chỉ vẽ phần nhìn thấy, không đòi vật phải trọn ảnh | Không coi vật bị cắt là lý do bỏ nhãn. Cần xác nhận class theo taxonomy trước khi bổ sung; chưa sửa phần này. |
| `cp4_curb/7d83710e-4697c3b2.jpg`, nắp capo ở mép dưới và ranh đường–vỉa hè | Gán road theo màu gần mặt đường / để ngoài taxonomy vì là thân xe | Overlay có màu road trên một phần nắp capo; task chỉ gồm road và sidewalk. Ranh phải dựa chức năng, không chỉ màu | Phần capo không nên là road; lỗi còn tồn tại, chưa sửa/export lại checkpoint. Không tô ép vùng ngoài taxonomy. |
| `hard_panoptic/000000460147.jpg`, xe chở ô tô ở giữa phía dưới ảnh | Gộp cả xe chở và các ô tô trên đó thành một vật / tách từng xe nhìn thấy | Ảnh có nhiều thân xe xếp trên xe vận chuyển; panoptic yêu cầu mỗi thing là instance riêng, chỉ gán phần nhìn thấy | Nguyên tắc là tách từng vật có đủ bằng chứng; không nối mask xuyên phần che. Chưa xác minh thủ công mọi ID trên xe vận chuyển; đề nghị coach góp ý cho phần biên/định danh khó phân biệt. |

Nguồn mô hình: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), [Segment Anything](https://github.com/facebookresearch/segment-anything), [SegFormer-B0 Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024), [Mask2Former COCO](https://huggingface.co/facebook/mask2former-swin-small-coco-panoptic). Ảnh và taxonomy dùng từ repo đề bài. Ground truth, cache mô hình, mật khẩu và token không đưa lên fork.
