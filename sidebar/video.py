from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
import os
import pandas as pd
import threading
cv2.setNumThreads(1) 

class VideoSearchApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.target_image = None
        self.video_path = None
        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        self.setWindowTitle("Video Search Application")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Buttons
        self.upload_button = QtWidgets.QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)

        self.search_button = QtWidgets.QPushButton("Search Target Image")
        self.search_button.clicked.connect(self.start_search)
        layout.addWidget(self.search_button)

        # Video display area
        self.video_label = QtWidgets.QLabel("Video will be displayed here")
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)

        # Result display area
        self.result_label = QtWidgets.QLabel("Search results will be displayed here")
        layout.addWidget(self.result_label)

    def upload_video(self):
        """Handle video file upload."""
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if self.video_path:
            self.result_label.setText(f"Video uploaded: {os.path.basename(self.video_path)}")
        else:
            self.result_label.setText("No video selected.")

    def start_search(self):
        """Start the search in a new thread and play video."""
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "No Video", "Please upload a video before starting the search.")
            return
        
        # Load target image
        target_image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Target Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not target_image_path:
            QtWidgets.QMessageBox.warning(self, "No Image", "Please select a target image.")
            return

        self.target_image = cv2.imread(target_image_path)
        if self.target_image is None:
            QtWidgets.QMessageBox.warning(self, "Invalid Image", "The selected target image is invalid.")
            return

        self.result_label.setText("Searching...")

        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Failed to Open Video", "Failed to open the video file.")
            return

        # Start the video playback and search
        self.timer.start(30)  # 30 ms interval for frame updates

        # Start the search in a separate thread
        search_thread = threading.Thread(target=self.search_target)
        search_thread.start()

    def update_frame(self):
        """Update the video frame displayed in QLabel."""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame to RGB format and display it in QLabel
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame_rgb.shape
                qimg = QtGui.QImage(frame_rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap)

            else:
                self.timer.stop()
                self.cap.release()

    def search_target(self):
        """Search for the target image in the video."""
        target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        target_w, target_h = self.target_image.shape[1], self.target_image.shape[0]

        frame_count = 0
        detected_frames = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(frame_gray, target_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.8:  # Detection threshold
                detected_frames.append(frame_count)

            frame_count += 1

        self.cap.release()

        if detected_frames:
            self.result_label.setText(f"Target image detected in frames: {', '.join(map(str, detected_frames))}")
        else:
            self.result_label.setText("Target image not detected in the video.")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = VideoSearchApp()
    window.show()
    sys.exit(app.exec_())
