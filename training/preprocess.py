import numpy
import pandas as pd
import numpy as np

def load_pose_data(dataset_path, action):
    df = pd.read_csv(dataset_path)
    video_grouped = df.groupby('video')

    video_data = []
    for video_name, group in video_grouped:
        video_frames = []

        for i in range(len(group)):
            frame_data = []

            for keypoint_id in range(24):
                x = group[f'x_{keypoint_id}'].iloc[i]
                y = group[f'y_{keypoint_id}'].iloc[i]
                z = group[f'z_{keypoint_id}'].iloc[i]
                # visibility = group[f'vis_{keypoint_id}'].iloc[i]
                # presence = group[f'pre_{keypoint_id}'].iloc[i]

                # frame_data.append([x, y, z, visibility, presence])
                frame_data.append([x, y, z])

            video_frames.append(np.array(frame_data))

        video_data.append(np.array(video_frames))

    split_index = int(len(video_data) * 0.8)
    train_data = video_data[:split_index]
    test_data = video_data[split_index:]

    if action == 0:
        train_labels = [[1, 0, 0] for _ in range(len(train_data))]
        test_labels = [[1, 0, 0] for _ in range(len(test_data))]
    elif action == 1:
        train_labels = [[0, 1, 0] for _ in range(len(train_data))]
        test_labels = [[0, 1, 0] for _ in range(len(test_data))]
    else:
        train_labels = [[0, 0, 1] for _ in range(len(train_data))]
        test_labels = [[0, 0, 1] for _ in range(len(test_data))]

    return train_data, train_labels, test_data, test_labels


def fill_with_last_data(sequence, desired_length):
    for i in range(len(sequence)):
        frames = sequence[i]
        if frames.shape[0] < desired_length:
            last_frame = frames[-1]

            frames = np.concatenate(
                [frames, np.repeat(last_frame[np.newaxis, ...], desired_length - frames.shape[0], axis=0)],
                axis=0
            )

            sequence[i] = frames

    return sequence


def process_dataset():
    (
        hand_raise_train,
        hand_raise_train_labels,
        hand_raise_test,
        hand_raise_test_labels
    ) = load_pose_data('../preprocess/mediapipe/dataset_csv/HandRaise.csv', 0)

    (
        sitting_train,
        sitting_train_labels,
        sitting_test,
        sitting_test_labels
    ) = load_pose_data('../preprocess/mediapipe/dataset_csv/Sitting.csv', 1)

    (
        writing_train,
        writing_train_labels,
        writing_test,
        writing_test_labels
    ) = load_pose_data('../preprocess/mediapipe/dataset_csv/Writing.csv', 2)

    train = hand_raise_train + sitting_train + writing_train
    train_labels = hand_raise_train_labels + sitting_train_labels + writing_train_labels

    test = hand_raise_test + sitting_test + writing_test
    test_labels = hand_raise_test_labels + sitting_test_labels + writing_test_labels

    max_frames_train = max([clip.shape[0] for clip in train])
    max_frames_test = max([clip.shape[0] for clip in test])
    max_frames = max(max_frames_train, max_frames_test)

    train = fill_with_last_data(train, max_frames)
    test = fill_with_last_data(test, max_frames)

    return numpy.array(train), numpy.array(train_labels), numpy.array(test), numpy.array(test_labels), max_frames
