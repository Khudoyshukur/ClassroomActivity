from traceback import print_last

import cv2
import numpy
import tensorflow as tf

from inference.MediapipePoseExtractor import MediapipePoseExtractor
from preprocess.mediapipe.util.utils import attach_landmark_to_frame

extractor = MediapipePoseExtractor()
cap = cv2.VideoCapture("../dataset/HandRaise/HandRaise_001.mov")
# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 24)
model = tf.keras.models.load_model('../training/pose_cnn_lstm_model.keras')

max_frames = 102
tracking_frames = []

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    pose_data = extractor.extract_pose_data(frame)
    if pose_data.pose_landmarks:
        tracking_frames.append(extractor.format_pose_data(pose_data))

        if len(tracking_frames) > max_frames:
            tracking_frames.pop(0)

        while len(tracking_frames) < max_frames:
            tracking_frames.append(tracking_frames[-1])

        input_sequence = numpy.array(tracking_frames)
        input_sequence = numpy.expand_dims(input_sequence, axis=0)

        prediction = model.predict(input_sequence)

        max_action = max(prediction[0][0], prediction[0][1], prediction[0][2])
        if max_action == prediction[0][0]:
            color = (0, 255, 0)  # Green
        elif max_action == prediction[0][1]:
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0) # Blue

        image_height, image_width, _ = frame.shape
        cv2.circle(frame, (image_width - 50, 50), 20, color, -1)

        frame = attach_landmark_to_frame(frame, pose_data)

    frame = cv2.flip(frame, 1)

    cv2.imshow('MediaPipe Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()