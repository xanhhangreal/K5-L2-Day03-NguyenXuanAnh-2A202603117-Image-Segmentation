# Lab 3 — Nhật ký chạy tự động

Cập nhật UTC+7: 2026-11-12T14:28:33.295262+07:00

Đã hoàn tất 8 task, 11 ảnh, 245 mask.

Dữ liệu được tạo tự động bằng YOLO + SAM + SegFormer. Kiểm tra cấu trúc không chứng minh mask đúng về nội dung.
Không thực hiện hoặc xác nhận bước tự vẽ/kiểm tra thủ công thay cho học viên. Không push GitHub hoặc nộp LMS.

## Cấu hình

```json
{
  "version": "lab3-auto-v1",
  "yolo": "yolo11n.pt",
  "imgsz": 1280,
  "conf": 0.25,
  "iou": 0.5,
  "semantic": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
  "sam": "SAM ViT-B CPU",
  "threads": 4
}
```

## Bảo toàn Easy

Easy không thay đổi: True

## Task và ZIP

| Task | Trạng thái | Task / Job | Masks | Giây | ZIP bytes |
|---|---|---|---:|---:|---:|
| cp2_slice | complete | 6 / 6 | 12 | 16.99 | 7677 |
| cp4_curb | complete | 7 / 7 | 2 | 17.21 | 19057 |
| medium_instance | complete | 17 / 17 | 104 | 27.0 | 69617 |
| hard_panoptic | complete | 16 / 16 | 73 | 27.24 | 78560 |
| cp1_holes | complete | 10 / 10 | 6 | 35.64 | 12088 |
| cp5_occlusion | complete | 11 / 11 | 38 | 38.57 | 25087 |
| cp3_thin | complete | 12 / 12 | 4 | 25.57 | 32513 |
| cp6_coverage | complete | 13 / 13 | 6 | 24.81 | 32008 |

## cp2_slice

Task: http://localhost:8080/tasks/6
ZIP: `submissions/cp2_slice.zip`
SHA-256: `4e6dffd1d0ab03f74032d895b0092d05adeadeae7b0d3d99f4babe8592f3185d`

- `000000017627.jpg`: 12 masks; 14.17 giây; chưa gán nhãn 279617 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 2 masks.
  Lớp: {"car": 9, "person": 3}
  - 2 masks have confidence below 0.5 (model-specific, not calibrated).

## cp4_curb

Task: http://localhost:8080/tasks/7
ZIP: `submissions/cp4_curb.zip`
SHA-256: `3fee2eb833c9a824dc7e9bb10593423916cb7bc2d081c6a1e8d6ebe7b19325e4`

- `7d83710e-4697c3b2.jpg`: 2 masks; 12.71 giây; chưa gán nhãn 479663 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 0 masks.
  Lớp: {"road": 1, "sidewalk": 1}

## medium_instance

Task: http://localhost:8080/tasks/17
ZIP: `submissions/medium_instance.zip`
SHA-256: `3f0ffe51f3a78fa9454619f6de4d00b36b55461732905722c393ae6ad67120df`
Mô hình của phiên bản hiện tại: ensemble-050

- `000000181542.jpg`: 36 masks; 0.23 giây; chưa gán nhãn 204415 pixel; chồng lấn trước xử lý 34278 pixel; confidence thấp 5 masks.
  Lớp: {"person": 23, "motorcycle": 7, "car": 4, "bus": 1, "bicycle": 1}

- `000000373353.jpg`: 32 masks; 0.14 giây; chưa gán nhãn 228604 pixel; chồng lấn trước xử lý 1198 pixel; confidence thấp 1 masks.
  Lớp: {"person": 11, "bus": 1, "car": 20}

- `000000458325.jpg`: 36 masks; 0.15 giây; chưa gán nhãn 190913 pixel; chồng lấn trước xử lý 10251 pixel; confidence thấp 3 masks.
  Lớp: {"person": 15, "car": 21}

## hard_panoptic

Task: http://localhost:8080/tasks/16
ZIP: `submissions/hard_panoptic.zip`
SHA-256: `10c20ad9b04a1f567a3fdb10ef50475dc8a59dd95b17963d7ac07ddc16ab6e83`
Mô hình của phiên bản hiện tại: tiled-segformer

- `000000350023.jpg`: 36 masks; 40.24 giây; chưa gán nhãn 11764 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 6 masks.
  Lớp: {"car": 22, "bus": 2, "traffic light": 2, "person": 4, "motorcycle": 1, "road": 1, "sidewalk": 1, "building": 1, "vegetation": 1, "sky": 1}

- `000000460147.jpg`: 37 masks; 49.41 giây; chưa gán nhãn 44080 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 9 masks.
  Lớp: {"truck": 9, "car": 23, "road": 1, "sidewalk": 1, "building": 1, "vegetation": 1, "sky": 1}
- Cảnh báo kiểm ZIP: COCO 1.0 chỉ xác nhận các mask; hãy kiểm trực quan chồng lấn và phủ vùng panoptic trong CVAT

## cp1_holes

Task: http://localhost:8080/tasks/10
ZIP: `submissions/cp1_holes.zip`
SHA-256: `b57f8d1c05fb5e244c28d8b67cb9f547b13100e02332bda66806ab1b3f7a3b68`

- `000000144300.jpg`: 6 masks; 10.22 giây; chưa gán nhãn 159766 pixel; chồng lấn trước xử lý 4305 pixel; confidence thấp 2 masks.
  Lớp: {"person": 3, "motorcycle": 1, "bus": 1, "car": 1}
  - 2 masks have confidence below 0.5 (model-specific, not calibrated).

## cp5_occlusion

Task: http://localhost:8080/tasks/11
ZIP: `submissions/cp5_occlusion.zip`
SHA-256: `2d41b0bd5668aa87b839cd6752829bdfe132d1fa8605959e0844e4f3f1f8c16b`

- `000000336232.jpg`: 38 masks; 12.13 giây; chưa gán nhãn 156064 pixel; chồng lấn trước xử lý 13495 pixel; confidence thấp 17 masks.
  Lớp: {"car": 29, "person": 2, "motorcycle": 1, "truck": 4, "bus": 2}
  - 17 masks have confidence below 0.5 (model-specific, not calibrated).

## cp3_thin

Task: http://localhost:8080/tasks/12
ZIP: `submissions/cp3_thin.zip`
SHA-256: `0613dc68ea14a2afb89c7ae0ed686f4e5845c049be116c3fce40303182c7e621`

- `839f7736-abe28069.jpg`: 4 masks; 0.29 giây; chưa gán nhãn 353468 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 0 masks.
  Lớp: {"road": 1, "pole": 1, "traffic sign": 1, "sky": 1}

## cp6_coverage

Task: http://localhost:8080/tasks/13
ZIP: `submissions/cp6_coverage.zip`
SHA-256: `a8fcf6f83bd64dd35baee99ddfcbc9d3b6866e7a6f58190a403a8ab2052fd540`

- `7daa6479-67988f3f.jpg`: 6 masks; 0.29 giây; chưa gán nhãn 62323 pixel; chồng lấn trước xử lý 0 pixel; confidence thấp 0 masks.
  Lớp: {"road": 1, "sidewalk": 1, "building": 1, "vegetation": 1, "person": 1, "car": 1}

## Quan sát overlay bởi trợ lý (không phải kiểm thủ công của học viên)

- cp2_slice / 000000017627.jpg: các xe sát nhau có mask riêng; phần xe lớn bị cắt ở mép trái chưa thấy mask trong overlay.
- cp4_curb / 7d83710e-4697c3b2.jpg: vùng đường/vỉa hè đã được phân biệt; một phần nắp capo ở mép dưới có màu road, cần sửa nếu dùng làm bài nộp.

## Kiểm thử tool

```json
{
  "core_tests": {
    "failed": 0,
    "cases": [
      "RLE empty/full/disconnected round-trip",
      "Invalid RLE rejection",
      "CVAT reordering and float serialization",
      "Panoptic precedence and taxonomy exclusion"
    ],
    "passed": 4
  },
  "pilot_exports": "passed",
  "pilot_tasks": [
    "cp2_slice",
    "cp4_curb"
  ],
  "rerun_identical_ids_annotations_zips": true,
  "all_zip_structure_checks": "9 passed, 0 errors, 0 missing",
  "launcher": "PowerShell full rerun passed using installed Docker image"
}

```

## Phiên bản thư viện đã chạy

```json
{
  "ultralytics": "8.4.210",
  "transformers": "4.57.6",
  "torch": "2.14.1+cpu",
  "torchvision": "0.29.2+cpu",
  "docker": "7.2.0",
  "requests": "2.34.2"
}
```

## Giới hạn

- Vùng không thuộc taxonomy được để trống, không bịa nhãn để đủ coverage.
- Vật nhỏ, biên mảnh, vật bị che và kính/khe cần kiểm trực quan; không tự suy ra phần khuất.
- Overlay nằm trong `automation_previews/`. Báo cáo này không thay báo cáo cá nhân theo mẫu lớp.
