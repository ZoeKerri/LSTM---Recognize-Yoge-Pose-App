import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import os
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from collections import deque
import uuid
import time
import pygame.mixer # Import pygame.mixer for audio

# Assume ExerciseCounter is defined elsewhere or will be provided.
# For this example, I'll provide a minimal placeholder if it's not defined in the user's main code.
try:
    from exercise_counter import ExerciseCounter
except ImportError:
    class ExerciseCounter:
        def __init__(self):
            self.exercise_counter = {
                "Garland_Pose": 0, "Happy_Baby_Pose": 0, "Head_To_Knee_Pose": 0, "Lunge_Pose": 0,
                "Mountain_Pose": 0, "Plank_Pose": 0, "Raised_Arms_Pose": 0, "Seated_Forward_Bend": 0,
                "Staff_Pose": 0, "Standing_Forward_Bend": 0
            }
            self.last_predicted_class = None
            self.start_time = None
            self.frame_count = 0

        def counting(self, predicted_class, pose_landmarks):
            notification = ""
            current_pose_time = 0

            if predicted_class == "Unknown":
                self.last_predicted_class = None
                self.start_time = None
                self.frame_count = 0
                return "You are doing it wrong", "Unknown"

            if self.last_predicted_class == predicted_class:
                self.frame_count += 1
                if self.start_time is None:
                    self.start_time = time.time()
                current_pose_time = int(time.time() - self.start_time)
            else:
                if self.last_predicted_class:
                    self.exercise_counter[self.last_predicted_class] += self.frame_count // 30
                self.last_predicted_class = predicted_class
                self.start_time = time.time()
                self.frame_count = 1
                current_pose_time = 0

            notification = f"Current pose: {predicted_class}. Time: {current_pose_time}s"
            return notification, predicted_class


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(-1, 79, 281, 671))
        self.widget.setStyleSheet("background-color: white")
        self.widget.setObjectName("widget")
        
        self.exercise_labels = {}
        self.exercises = [
            "Garland_Pose", "Happy_Baby_Pose", "Head_To_Knee_Pose", "Lunge_Pose",
            "Mountain_Pose", "Plank_Pose", "Raised_Arms_Pose", "Seated_Forward_Bend",
            "Staff_Pose", "Standing_Forward_Bend"
        ]
        
        for i, exercise in enumerate(self.exercises):
            label = QtWidgets.QLabel(parent=self.widget)
            label.setGeometry(QtCore.QRect(0, 100 + i * 50, 281, 31))
            font = QtGui.QFont()
            font.setFamily("Times New Roman")
            font.setPointSize(14)
            label.setFont(font)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setObjectName(f"label_{exercise}")
            self.exercise_labels[exercise] = label
        
        self.refreshbutton = QtWidgets.QPushButton(parent=self.widget)
        self.refreshbutton.setGeometry(QtCore.QRect(60, 600, 151, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.refreshbutton.setFont(font)
        self.refreshbutton.setStyleSheet("background-color: black; color: white")
        self.refreshbutton.setObjectName("refreshbutton")
        
        self.labelTitle = QtWidgets.QLabel(parent=self.centralwidget)
        self.labelTitle.setGeometry(QtCore.QRect(0, 0, 1211, 79))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.labelTitle.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        font.setBold(True)
        self.labelTitle.setFont(font)
        self.labelTitle.setStyleSheet("color: white; background-color: black")
        self.labelTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.labelTitle.setObjectName("labelTitle")
        
        self.widget_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(280, 80, 931, 671))
        self.widget_2.setStyleSheet("background-color: #CCCCCC")
        self.widget_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.widget_2.setObjectName("widget_2")

        self.standard_img_label = QtWidgets.QLabel(parent=self.widget_2)
        self.standard_img_label.setGeometry(QtCore.QRect(self.widget_2.width() - 210, self.widget_2.height() - 160, 200, 150))
        self.standard_img_label.setStyleSheet("background-color: rgba(0, 0, 0, 100); border: 2px solid white;")
        self.standard_img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.standard_img_label.setObjectName("standard_img_label")
        self.standard_img_label.lower()
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI Yoga Pose Counter"))
        for exercise, label in self.exercise_labels.items():
            label.setText(_translate("MainWindow", f"{exercise.replace('_', ' ')}: 0"))
        self.refreshbutton.setText(_translate("MainWindow", "Làm mới"))
        self.labelTitle.setText(_translate("MainWindow", "Hãy tập các tư thế yoga"))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_counter_signal = pyqtSignal(dict)
    notification_signal = pyqtSignal(str)
    update_standard_image_signal = pyqtSignal(str)

    def __init__(self, exercises_list): 
        super().__init__()
        self._run_flag = True
        self.exercises = exercises_list 

        model_preproc_dir = "" 
        
        try:
            model_path = os.path.join(model_preproc_dir, "best_yoga_pose_model.h5") if model_preproc_dir else "best_yoga_pose_model.h5"
            scaler_path = os.path.join(model_preproc_dir, "scaler.pkl") if model_preproc_dir else "scaler.pkl"
            encoder_path = os.path.join(model_preproc_dir, "label_encoder.pkl") if model_preproc_dir else "label_encoder.pkl"

            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            print("VideoThread: Đã tải mô hình, scaler, và encoder thành công!")
        except Exception as e:
            print(f"VideoThread: Lỗi khi tải các file mô hình/tiền xử lý: {e}")
            print("VideoThread: Đảm bảo các file mô hình nằm đúng vị trí.")
            sys.exit(1)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                      min_detection_confidence=0.7, 
                                      min_tracking_confidence=0.7)
        self.selected_landmarks = list(range(33))
        self.exercise_counter = ExerciseCounter()
        self.current_exercise = "None"
        self.frame_buffer = deque(maxlen=50)
        self.low_confidence_count = 0
        self.confidence_threshold = 0.8
        self.frame_threshold = 10
        self.keypoint_threshold = 20
        self.visibility_threshold = 0.5     

        # --- CẤU HÌNH ÂM THANH GỢI Ý ---
        pygame.mixer.init()
        self.hint_audio_path = "Hint_Audio" # Thư mục chứa các file âm thanh
        self.hint_audios = {}
        self.last_hint_play_time = 0 # Thời điểm cuối cùng phát gợi ý
        self.hint_cooldown = 5 # Giây: Thời gian chờ giữa các lần phát gợi ý

        # Tải các file âm thanh gợi ý
        if os.path.exists(self.hint_audio_path):
            for exercise in self.exercises: # Lặp qua danh sách bài tập đã truyền vào
                audio_file_mp3 = os.path.join(self.hint_audio_path, f"{exercise}.mp3")
                
                if os.path.exists(audio_file_mp3):
                    try:
                        self.hint_audios[exercise] = pygame.mixer.Sound(audio_file_mp3)
                    except pygame.error as e:
                        print(f"Warning: Could not load MP3 audio for {exercise} ({audio_file_mp3}): {e}")
                else:
                    print(f"Warning: No audio file (.mp3 or .wav) found for {exercise} in {self.hint_audio_path}")
        else:
            print(f"Warning: Hint_Audio folder '{self.hint_audio_path}' not found. Audio hints will not play.")

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                self.change_pixmap_signal.emit(processed_frame)
            else:
                continue
        cap.release()

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        current_predicted_exercise = "Unknown" 
        notification_message = "Please ensure your body is in the camera"
        confidence_display = 0.0
        visible_keypoints = 0

        if results.pose_landmarks:
            visible_keypoints = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > self.visibility_threshold)
            
            if visible_keypoints < self.keypoint_threshold:
                current_predicted_exercise = "Unknown"
                self.low_confidence_count = 0
                self.frame_buffer.clear()
            else:
                row = []
                for i in self.selected_landmarks:
                    landmark = results.pose_landmarks.landmark[i]
                    row.extend([landmark.x, landmark.y, landmark.z])

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                
                input_data = np.array(row).reshape(1, -1)
                input_data_scaled = self.scaler.transform(input_data)
                self.frame_buffer.append(input_data_scaled[0])

                if len(self.frame_buffer) == 50:
                    input_sequence = np.array(self.frame_buffer).reshape(1, 50, 99)
                    prediction = self.model.predict(input_sequence, verbose=0)[0]
                    
                    predicted_index = np.argmax(prediction)
                    confidence_display = prediction[predicted_index]

                    predicted_class = self.encoder.inverse_transform([predicted_index])[0]

                    if confidence_display < self.confidence_threshold:
                        self.low_confidence_count += 1
                        if self.low_confidence_count >= self.frame_threshold:
                            notification_message = "You are doing it wrong"
                            # Khi đang làm sai, current_exercise vẫn là bài tập mà mô hình nghĩ, không phải "Unknown"
                            current_predicted_exercise = predicted_class
                            self.low_confidence_count = self.frame_threshold # Giữ cho bộ đếm không vượt quá giới hạn

                            # Phát âm thanh gợi ý
                            if current_predicted_exercise in self.hint_audios and \
                               not pygame.mixer.get_busy() and \
                               (time.time() - self.last_hint_play_time > self.hint_cooldown):
                                self.hint_audios[current_predicted_exercise].play()
                                self.last_hint_play_time = time.time()
                                
                        else:
                            notification_message, temp_exercise = self.exercise_counter.counting(predicted_class, results.pose_landmarks)
                            current_predicted_exercise = temp_exercise
                    else:
                        self.low_confidence_count = 0
                        notification_message, temp_exercise = self.exercise_counter.counting(predicted_class, results.pose_landmarks)
                        current_predicted_exercise = temp_exercise
                else:
                    current_predicted_exercise = "Collecting data..."
                    notification_message = "Please hold still for pose recognition."

        else: # No pose landmarks detected
            current_predicted_exercise = "Unknown"
            self.low_confidence_count = 0
            self.frame_buffer.clear()
            notification_message = "Please ensure your body is in the camera"

        # Update current_exercise và gửi tín hiệu
        if self.current_exercise != current_predicted_exercise:
            self.current_exercise = current_predicted_exercise
            self.update_standard_image_signal.emit(self.current_exercise)

        self.update_counter_signal.emit(self.exercise_counter.exercise_counter)
        self.notification_signal.emit(notification_message)
        
        # Hiển thị thông tin trên frame (loại bỏ dấu gạch dưới)
        cv2.putText(frame, f"Current: {self.current_exercise.replace('_', ' ')}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"{notification_message.replace('_', ' ')}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Confidence: {confidence_display:.2%} | Keypoints: {visible_keypoints}", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.exercise_counts = {
            "Garland_Pose": 0, "Happy_Baby_Pose": 0, "Head_To_Knee_Pose": 0, "Lunge_Pose": 0,
            "Mountain_Pose": 0, "Plank_Pose": 0, "Raised_Arms_Pose": 0, "Seated_Forward_Bend": 0,
            "Staff_Pose": 0, "Standing_Forward_Bend": 0
        }
        
        self.standard_image_path = "Standard_Img"
        self.standard_pixmaps = {}
        self.load_standard_images()

        # Truyền danh sách bài tập vào VideoThread
        self.video_thread = VideoThread(self.exercises) 
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_counter_signal.connect(self.update_counts)
        self.video_thread.notification_signal.connect(self.update_notification)
        self.video_thread.update_standard_image_signal.connect(self.update_standard_image_display)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_labels)
        self.timer.start(100)
        
        self.refreshbutton.clicked.connect(self.reset_all)
        self.video_thread.start()

    def load_standard_images(self):
        for exercise_name in self.exercises:
            image_file = os.path.join(self.standard_image_path, f"{exercise_name}.png")
            if os.path.exists(image_file):
                pixmap = QPixmap(image_file)
                scaled_pixmap = pixmap.scaled(self.standard_img_label.size(), 
                                              Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                self.standard_pixmaps[exercise_name] = scaled_pixmap
            else:
                print(f"Warning: Standard image not found for {exercise_name} at {image_file}")
                self.standard_pixmaps[exercise_name] = QPixmap()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.widget_2.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.widget_2.width(), self.widget_2.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_labels(self, current_exercise_name=None):
        for exercise, label in self.exercise_labels.items():
            seconds = self.exercise_counts[exercise]
            minutes = seconds // 60
            seconds = seconds % 60
            label.setText(f"{exercise.replace('_', ' ')}: {minutes:02d}:{seconds:02d}")
            if exercise == self.video_thread.current_exercise:
                label.setStyleSheet("color: red;")
            else:
                label.setStyleSheet("color: black;")
    
    def update_counts(self, new_counts):
        self.exercise_counts = new_counts
        self.update_labels()

    def update_notification(self, message):
        self.labelTitle.setText(message if message else "Hãy tập các tư thế yoga")

    def update_standard_image_display(self, exercise_name):
        if exercise_name in self.standard_pixmaps and self.standard_pixmaps[exercise_name]:
            self.standard_img_label.setPixmap(self.standard_pixmaps[exercise_name])
        elif exercise_name == "Unknown" or exercise_name == "Collecting data...":
            self.standard_img_label.clear()
            self.standard_img_label.setText("No Standard Pose") 
        else:
            self.standard_img_label.clear()
            self.standard_img_label.setText("Loading...")

    def reset_all(self):
        for exercise in self.exercise_counts:
            self.exercise_counts[exercise] = 0
        self.update_labels()
        self.video_thread.exercise_counter = ExerciseCounter()
        self.video_thread.frame_buffer.clear()
        self.video_thread.low_confidence_count = 0
        self.labelTitle.setText("Hãy tập các tư thế yoga")
        self.standard_img_label.clear()
        self.standard_img_label.setText("No Standard Pose")


    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
