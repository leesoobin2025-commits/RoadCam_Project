# RoadCam
M2 - M8
Phần kéo tương phản chưa dừng khi a >= b
NODE 30/10
- Thêm các chức năng hiệu chỉnh log và cân bằng histogram
    +Chưa thêm so sánh
- Sửa lại ROI để bấm chuột sẽ dừng việc cắt ROI ( tường minh hành động):
    + Thêm self.is_dragging = False ở ImageCaves.__init__() và set_interaction_mode()
    + Sửa phần mousePressEvent()
    + mouseMoveEven() sửa điều kiện cuối cùng if self.interaction_mode == "roi" and self.is_dragging:
    + Sửa mouseReleaseEvent()

Phần cần sửa và lưu ý
- Histogram chưa kiểm tra ảnh xám

#Đã xem video được nhưng vẫn bị lỗi không chỉnh ROI áp ROI và điều khiển M7 M5
#phát hiện làn trên video, phát hiện chuyển động\
#Tóm lại phần M8 chưa liên kết được với M7 và M5

#Phân ROI trong đoạn run_command có 1 đoạn để ROI mà vẫn hiển thị ảnh nền (kêu KHÁNH đọc hiểu)

chỗ M8 apply_video_overlay có đoạn 
config = getattr(self, "current_operation_config", lambda: None)()
        if config is not None and getattr(self, "preview_button", None) and self.preview_button.isChecked():
            try:
                return self.run_command(config, frame, self.current_params())
            except (cv2.error, ValueError, TypeError, ZeroDivisionError) as error:
                self.status_message.setText(f"Không áp dụng được lên video: {error}")
        return frame.copy()
Main_interface sua 
def schedule_preview(self, *_):
    if self.current_image is None:
                return
            if not self.is_video_open or not self.video_timer.isActive():
                self.preview_timer.start()
def update_preview(self):
        # Đoạn thêm
        if self.is_video_open:
            self.refresh_video_frame()
            return
        .....