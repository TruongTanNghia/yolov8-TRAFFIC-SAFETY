# Dataset và huấn luyện — kế hoạch, chưa có dữ liệu

Repo hiện chưa có dataset hay nhãn. Không tải và huấn luyện model thay thế khi chưa xác minh giấy phép.

## Quy trình dữ liệu

1. Lập danh mục nguồn công khai hoặc video tự thu thập có quyền dùng; lưu nguồn, ngày, giấy phép và giới hạn phân phối.
2. Gán nhóm đối tượng `person`, `motorcycle`, `helmet` và nhãn trạng thái đầu `helmet_on`, `helmet_off`, `uncertain`. Định nghĩa người điều khiển/ngồi sau riêng nếu dữ liệu đủ quan sát; không suy diễn từ một box người bất kỳ.
3. Đánh dấu `uncertain` khi vùng đầu nhỏ, bị che, ngược sáng hoặc góc camera không thể phân định. Không buộc nhãn nhị phân.
4. Chia train/validation/test theo camera hoặc phiên ghi hình, không chia frame ngẫu nhiên của cùng clip sang nhiều split.
5. Báo cáo cân bằng nhãn, độ phân giải head ROI và nhóm điều kiện ánh sáng/góc nhìn; rà soát nhầm lẫn bằng người gán nhãn thứ hai.
6. Ghi config, seed, phiên bản dependency, commit dữ liệu, log, checkpoint tốt nhất, confusion matrix và phân tích lỗi cho mỗi experiment.

## Chỉ số

Đánh giá detector bằng precision/recall/F1/mAP và false positives/negatives theo lớp; tracker bằng IDF1/HOTA trên sequence gán ID; nghiệp vụ bằng precision/recall event, duplicate rate và tỷ lệ `uncertain`. FPS/latency/RAM/VRAM đo trên thiết bị triển khai. Chưa có số đo trong repository.
