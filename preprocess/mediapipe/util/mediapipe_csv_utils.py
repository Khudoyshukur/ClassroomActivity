import os
import csv
from preprocess.mediapipe.util.VideoLandmarkResult import VideoLandmarkResult

def get_mediapipe_csv_dataset_labels():
    result = ["action", "video"]

    for i in range(33):
        result.append(f"x_{i}")
        result.append(f"y_{i}")
        result.append(f"z_{i}")
        result.append(f"vis_{i}")
        result.append(f"pre_{i}")

    return result


def get_csv_rows(result: VideoLandmarkResult):
    rows_result = []

    for landmarkResult in result.landmarks:
        single_human_landmarks = landmarkResult.pose_landmarks[0]
        length = len(single_human_landmarks)

        row = [result.action, result.video_name]
        for index in range(length):
            landmark = single_human_landmarks[index]

            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)
            row.append(landmark.visibility)
            row.append(landmark.presence)

        rows_result.append(row)

    return rows_result

def write_to_csv(folder, filename, landmark_results):
    os.makedirs(folder, exist_ok=True)

    csv_data = [get_mediapipe_csv_dataset_labels()]

    for result in landmark_results:

        rows = get_csv_rows(result)
        for row in rows:
            csv_data.append(row)

    with open(f"{folder}/{filename}", 'w', newline='\n') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_data)