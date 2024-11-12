import os

from preprocess.mediapipe.util.MediapipePoseExtractor import MediapipePoseExtractor
from preprocess.mediapipe.util.mediapipe_csv_utils import write_to_csv
from preprocess.mediapipe.util.utils import get_video_name_from_path
import warnings

warnings.simplefilter('ignore', category=Warning)

handRaiseFolder = "../../dataset_csv/HandRaise"
sittingFolder = "../../dataset_csv/Sitting"
writingFolder = "../../dataset_csv/Writing"

extractor = MediapipePoseExtractor("model/pose_landmarker_heavy.task")

def extract_pose_data(action, path):
    files = os.listdir(path)
    file_paths = [os.path.join(path, file) for file in files]
    file_paths.sort(key=os.path.getmtime, reverse=False)

    result = []
    for file_path in file_paths:
        print(f"Processing --> {action} --> {get_video_name_from_path(file_path)}")
        result.append(extractor.extract_video_pose_data(action, file_path))

    write_to_csv("dataset_csv", f"{action}.csv", result)

extract_pose_data("HandRaise", handRaiseFolder)
extract_pose_data("Sitting", sittingFolder)
extract_pose_data("Writing", writingFolder)