import mediapipe as mp
import numpy
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerResult

class MediapipePoseExtractor:
    def __init__(self):
        self.base_options = mp.tasks.BaseOptions(model_asset_path="../preprocess/mediapipe/model/pose_landmarker_heavy.task")
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        pass

    def extract_pose_data(self, frame) -> PoseLandmarkerResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect(image)

    def format_pose_data(self, result):
        arr = []

        single_person = result.pose_landmarks[0]
        for index in range(24):
            landmark = single_person[index]
            # arr.append([landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence])
            arr.append([landmark.x, landmark.y, landmark.z])

        return numpy.array(arr)