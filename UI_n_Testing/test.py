import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import os
import sys

video_to_process_path = "test_video/Mountain_Pose_test/2.mp4"
model_preproc_dir = ""

try:
    model_path = os.path.join(model_preproc_dir, "best_yoga_pose_model.h5") if model_preproc_dir else "best_yoga_pose_model.h5"
    scaler_path = os.path.join(model_preproc_dir, "scaler.pkl") if model_preproc_dir else "scaler.pkl"
    encoder_path = os.path.join(model_preproc_dir, "label_encoder.pkl") if model_preproc_dir else "label_encoder.pkl"

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    print("Đã tải mô hình, scaler, và encoder thành công!")
except Exception as e:
    sys.exit(1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

selected_landmarks = [i for i in range(0, 33)]

print(f"\nĐang xử lý video: {video_to_process_path}")

cap = cv2.VideoCapture(video_to_process_path)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở video '{video_to_process_path}'. Kiểm tra đường dẫn hoặc codec.")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30 

fixed_width = 800
fixed_height = int(fixed_width * frame_height / frame_width)

SEQUENCE_LENGTH = 50
sequence_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_display = cv2.resize(frame, (fixed_width, fixed_height))
    image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        row = []
        for i in selected_landmarks:
            landmark = results.pose_landmarks.landmark[i]
            row.extend([landmark.x, landmark.y, landmark.z])

        input_data_single_frame = np.array(row).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_single_frame)
        sequence_buffer.append(input_data_scaled[0])

        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        if len(sequence_buffer) == SEQUENCE_LENGTH:
            model_input = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, -1)
            predictions = model.predict(model_input, verbose=0)[0]
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_labels = encoder.inverse_transform(top_3_indices)
            top_3_probs = predictions[top_3_indices]

            for idx, (label, prob) in enumerate(zip(top_3_labels, top_3_probs)):
                cv2.putText(frame_display, f"{label}: {prob:.2%}", (30, 50 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_display, "Khong tim thay nguoi", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        sequence_buffer.clear()

    cv2.imshow("Video Prediction", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã hoàn tất xử lý video.")