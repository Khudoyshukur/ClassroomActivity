import os
import cv2
import mediapipe as mp
from preprocess.mediapipe.util.utils import attach_landmark_to_frame, get_video_name_from_path
from preprocess.mediapipe.util.VideoLandmarkResult import VideoLandmarkResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerResult

class MediapipePoseExtractor:
    def __init__(self, model_path):
        self.base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )

    def extract_video_pose_data(self, action: str, video_path: str) -> VideoLandmarkResult:
        cv = cv2.VideoCapture(video_path)

        video_name = get_video_name_from_path(video_path)
        folder_name = f"raw/{action}/{video_name}"
        os.makedirs(folder_name, exist_ok=True)

        video_landmarks = []
        while True:
            success, frame = cv.read()

            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = int(cv.get(cv2.CAP_PROP_POS_MSEC))
            result = self.extract_raw_pose(timestamp, image_rgb)

            frame_filename = os.path.join(folder_name, f'{timestamp}.jpg')

            if result.pose_landmarks:
                img = attach_landmark_to_frame(frame, result)
                cv2.imwrite(frame_filename, img)
                video_landmarks.append(result)
        cv.release()

        return VideoLandmarkResult(action, video_name, video_landmarks)

    def extract_raw_pose(self, timestamp, cv_frame) -> PoseLandmarkerResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_frame)

        landmarker = PoseLandmarker.create_from_options(self.options)
        return landmarker.detect_for_video(image, timestamp)
