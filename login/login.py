from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import subprocess
import cv2
import face_recognition
import numpy as np
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 772)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_menu = QtWidgets.QWidget(self.centralwidget)
        self.main_menu.setGeometry(QtCore.QRect(270, 0, 521, 751))
        self.main_menu.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.main_menu.setObjectName("main_menu")

        self.pushButton = QtWidgets.QPushButton(self.main_menu)
        self.pushButton.setGeometry(QtCore.QRect(220, 640, 111, 41))
        self.pushButton.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a68a6;
            }
        """)
        self.pushButton.setCheckable(True)
        self.pushButton.setAutoExclusive(True)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.main_menu)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 70, 191, 131))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/eye-removebg-preview.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon)
        self.pushButton_2.setIconSize(QtCore.QSize(100, 100))
        self.pushButton_2.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 0, 0);
                border: none;
                padding: 0px;
            }
        """)
        self.pushButton_2.setObjectName("pushButton_2")

        self.lineEdit = QtWidgets.QLineEdit(self.main_menu)
        self.lineEdit.setGeometry(QtCore.QRect(130, 20, 291, 41))
        self.lineEdit.setStyleSheet("color: rgb(255, 255, 255); font: 18pt 'MS Shell Dlg 2';")
        self.lineEdit.setObjectName("lineEdit")

        self.image_box = QtWidgets.QLabel(self.main_menu)
        self.image_box.setGeometry(QtCore.QRect(60, 200, 431, 361))
        self.image_box.setStyleSheet("""
            QLabel {
                border: 2px solid #4a90e2;
                border-radius: 5px;
                background-color: #333333;
            }
        """)
        self.image_box.setObjectName("image_box")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Start the camera feed initially
        self.face_recognition_thread = None
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image_box)
        self.camera_thread.start()

        # Connect the LOGIN button to start face recognition
        self.pushButton.clicked.connect(self.start_face_recognition)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "LOGIN"))
        self.lineEdit.setText(_translate("MainWindow", "EYE OF THE SAURON"))

    def update_image_box(self, qt_img):
        self.image_box.setPixmap(qt_img)

    def start_face_recognition(self):
        if self.face_recognition_thread is None or not self.face_recognition_thread.isRunning():
            self.face_recognition_thread = FaceRecognitionThread()
            self.face_recognition_thread.change_pixmap_signal.connect(self.update_image_box)
            self.face_recognition_thread.start()
        else:
            print("Face recognition is already running.")

import login_rc as login_rc

from PyQt5.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.camera_index = 0 #//camera index 
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera feed with index: {self.camera_index}")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(QPixmap.fromImage(qt_img))

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

class FaceRecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.target_image_path = "target.png"
        if not os.path.isfile(self.target_image_path):
            raise ValueError(f"Target image file does not exist: {self.target_image_path}")

        self.target_image = face_recognition.load_image_file(self.target_image_path)
        self.target_encodings = face_recognition.face_encodings(self.target_image)
        if not self.target_encodings:
            raise ValueError("No face encodings found in the target image")
        self.target_encoding = self.target_encodings[0]

        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera feed with index: {self.camera_index}")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
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
                print("Target person detected")
                subprocess.Popen(["python", "./sidebar/main.py"])
                self.cap.release()
                cv2.destroyAllWindows()
                QtWidgets.QApplication.quit()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(QPixmap.fromImage(qt_img))

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
