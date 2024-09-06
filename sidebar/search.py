import os
import shutil
import csv
import cv2
import face_recognition
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime

class FaceRecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)
    detection_signal = pyqtSignal(str)

    def __init__(self, image_path, person_name):
        super().__init__()
        self.image_path = image_path
        self.person_name = person_name
        self.target_image = face_recognition.load_image_file(self.image_path)
        self.target_encodings = face_recognition.face_encodings(self.target_image)
        if not self.target_encodings:
            raise ValueError("No face encodings found in the target image")
        self.target_encoding = self.target_encodings[0]

        self.camera_index = 0  # Adjust this index if needed
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera feed with index: {self.camera_index}")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from camera.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            detected = False
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = face_recognition.face_distance([self.target_encoding], face_encoding)
                if face_distances[0] < 0.6:
                    detected = True
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    break

            if detected:
                self.cap.release()
                cv2.destroyAllWindows()
                self.log_detection()
                self.detection_signal.emit(f"{self.person_name} detected")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(QPixmap.fromImage(qt_img))

        self.cap.release()
        cv2.destroyAllWindows()

    def log_detection(self):
        """Log the camera index, person name, and timestamp to a CSV file."""
        log_file = "detection_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.camera_index, self.person_name, timestamp,self.image_path])

class Ui_SearchApp(object):
    def setupUi(self, SearchApp):
        SearchApp.setObjectName("SearchApp")
        SearchApp.resize(800, 600)
        
        self.verticalLayout = QtWidgets.QVBoxLayout(SearchApp)
        SearchApp.setLayout(self.verticalLayout)
        
        self.image_label = QtWidgets.QLabel(SearchApp)
        self.image_label.setObjectName("image_label")
        self.image_label.setText("No image uploaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(800, 600)  # Set to size of camera feed
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.verticalLayout.addWidget(self.image_label)
        
        self.upload_button = QtWidgets.QPushButton(SearchApp)
        self.upload_button.setObjectName("upload_button")
        self.upload_button.setStyleSheet("""
        QPushButton {
            background-color: #4a90e2; /* Button background color */
            color: #ffffff; /* Button text color */
            border: none; /* Remove border */
            border-radius: 5px; /* Rounded corners */
            padding: 5px 10px; /* Smaller padding */
            font-size: 14px; /* Smaller text size */
            min-width: 30px; /* Minimum width */
            min-height: 30px; /* Minimum height */
        }
        QPushButton:hover {
            background-color: #357abd; /* Darker blue when hovering */
        }
        QPushButton:pressed {
            background-color: #003d6b; /* Even darker blue when pressed */
        }
        """)
        self.upload_button.setText("Upload Image")
        self.verticalLayout.addWidget(self.upload_button)

        self.name_input = QtWidgets.QLineEdit(SearchApp)
        self.name_input.setObjectName("name_input")
        self.name_input.setPlaceholderText("Enter person name")
        self.verticalLayout.addWidget(self.name_input)
        
        self.search_button = QtWidgets.QPushButton(SearchApp)
        self.search_button.setObjectName("search_button")
        self.search_button.setText("Start Search")
        self.verticalLayout.addWidget(self.search_button)
        
        self.upload_button.clicked.connect(self.upload_image)
        self.search_button.clicked.connect(self.start_search)

        self.image_path = None
        self.person_name = None
        self.target_folder = "target"
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

    def upload_image(self):
        """Handle the image upload and display it."""
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

            file_name = os.path.basename(file_path)
            target_path = os.path.join(self.target_folder, file_name)

            shutil.copy(file_path, target_path)
            self.upload_button.setEnabled(False)
            self.search_button.setEnabled(True)
        else:
            self.image_label.setText("No image uploaded")

    def start_search(self):
        """Start the image search in the camera feed."""
        if not self.image_path:
            QtWidgets.QMessageBox.warning(None, "No Image", "Please upload an image before starting the search.")
            return

        self.person_name = self.name_input.text()
        if not self.person_name:
            QtWidgets.QMessageBox.warning(None, "No Name", "Please enter a name before starting the search.")
            return
        
        QtWidgets.QMessageBox.information(None, "Search Started", "Searching for the image in the camera feed...")

        self.face_recognition_thread = FaceRecognitionThread(self.image_path, self.person_name)
        self.face_recognition_thread.change_pixmap_signal.connect(self.update_image_label)
        self.face_recognition_thread.detection_signal.connect(self.show_detection_message)
        self.face_recognition_thread.start()

    def update_image_label(self, pixmap):
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def show_detection_message(self, message):
        QtWidgets.QMessageBox.information(None, "Detection", message)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = Ui_SearchApp()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
