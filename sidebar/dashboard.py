from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import face_recognition
import numpy as np
from ultralytics import YOLO 

class Ui_Dashboard(object):
    def setupUi(self, Dashboard):
        Dashboard.setObjectName("Dashboard")
        Dashboard.resize(800, 600)
        
        self.verticalLayout = QtWidgets.QVBoxLayout(Dashboard)
        Dashboard.setLayout(self.verticalLayout)

        self.camera_widget1 = QtWidgets.QLabel(Dashboard)
        self.camera_widget1.setFixedSize(800, 400)
        self.camera_widget1.setStyleSheet("background-color: rgb(0, 0, 0); border: 2px solid #4a90e2; border-radius: 5px;")
        self.camera_widget1.setObjectName("camera_widget1")

        self.camera_widget2 = QtWidgets.QLabel(Dashboard)
        self.camera_widget2.setFixedSize(800, 400)
        self.camera_widget2.setStyleSheet("background-color: rgb(0, 0, 0); border: 2px solid #4a90e2; border-radius: 5px;")
        self.camera_widget2.setObjectName("camera_widget2")

        self.start_button = QtWidgets.QPushButton(Dashboard)
        self.start_button.setText("AI")
        self.start_button.setStyleSheet("background-color: #4a90e2; color: white;")
        self.start_button.setObjectName("start_button")

        self.verticalLayout.addWidget(self.camera_widget1)
        self.verticalLayout.addWidget(self.camera_widget2)
        self.verticalLayout.addWidget(self.start_button)

        self.camera_thread1 = CameraThread(0)  # Camera index 0
        self.camera_thread2 = CameraThread(1)  # Camera index 1
        self.camera_thread1.change_pixmap_signal.connect(self.update_camera_widget1)
        self.camera_thread2.change_pixmap_signal.connect(self.update_camera_widget2)

        self.camera_thread1.start()
        self.camera_thread2.start()

        self.start_button.clicked.connect(self.toggle_ai_processing)
        self.ai_processing_active = False

    def update_camera_widget1(self, qt_img):
        self.camera_widget1.setPixmap(qt_img)

    def update_camera_widget2(self, qt_img):
        self.camera_widget2.setPixmap(qt_img)

    def toggle_ai_processing(self):
        if self.ai_processing_active:
            self.camera_thread1.stop_ai_processing()
            self.camera_thread2.stop_ai_processing()
            self.start_button.setText("AI")
            print("AI processing stopped.")
        else:
            self.camera_thread1.start_ai_processing()
            self.camera_thread2.start_ai_processing()
            self.start_button.setText("Stop AI")
            print("AI processing started.")
        self.ai_processing_active = not self.ai_processing_active

class CameraThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QPixmap)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.yolo_model = YOLO("D:/Project-6th/pramod/project/yolo/best.pt")  # Load the YOLO model for face detection
        self.face_folder = "detected_faces"
        self.max_images_per_person = 5
        self.face_encodings = {}  
        self.face_id_map = {}  
        self.next_face_id = 0  
        self.similarity_threshold = 0.6  
        self.ai_processing = False

        if not os.path.exists(self.face_folder):
            os.makedirs(self.face_folder)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.ai_processing:
                results = self.yolo_model(frame)  # Perform face detection with YOLO
                current_face_ids = []
                for result in results:
                    # Bounding box format: [x1, y1, x2, y2]
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box.astype(int)
                        face_img = frame[y1:y2, x1:x2]
                        face_encoding = self.get_face_encoding(face_img)
                        
                        if face_encoding is not None:
                            face_encoding_tuple = tuple(face_encoding.flatten())
                            matched_id = self.find_face_id(face_encoding)
                            
                            if matched_id is not None:
                                person_folder = os.path.join(self.face_folder, f'person_{matched_id}')
                                if len(os.listdir(person_folder)) < self.max_images_per_person:
                                    self.save_face_image(face_img, matched_id)
                                current_face_ids.append(matched_id)
                            else:
                                new_id = self.next_face_id
                                person_folder = os.path.join(self.face_folder, f'person_{new_id}')
                                if not os.path.exists(person_folder):
                                    os.makedirs(person_folder)
                                self.save_face_image(face_img, new_id)
                                self.face_encodings[new_id] = face_encoding
                                self.face_id_map[face_encoding_tuple] = new_id
                                self.next_face_id += 1
                                current_face_ids.append(new_id)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f'ID: {new_id if matched_id is None else matched_id}', 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(QPixmap.fromImage(qt_img))

    def get_face_encoding(self, face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_img_rgb)
        return encodings[0] if encodings else None

    def find_face_id(self, face_encoding):
        for face_id, encoding in self.face_encodings.items():
            matches = face_recognition.compare_faces([encoding], face_encoding, tolerance=self.similarity_threshold)
            if matches[0]:
                return face_id
        return None

    def save_face_image(self, face_img, face_id):
        person_folder = os.path.join(self.face_folder, f'person_{face_id}')
        face_img_path = os.path.join(person_folder, f'face_{len(os.listdir(person_folder))}.jpg')
        cv2.imwrite(face_img_path, face_img)

    def start_ai_processing(self):
        self.ai_processing = True

    def stop_ai_processing(self):
        self.ai_processing = False

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
