"""
This will enable you to visualize the human pose of a tennis shot animated.
For this, you need to pass a csv file containing the features from extract_shots_as_features.py
"""

from argparse import ArgumentParser
import numpy as np
import cv2
import imageio
import pandas as pd

HEIGHT = 500
WIDTH = 500

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def draw_key_point(frame, shot_inst, key_point_str):
    """Draw key point (shoulders, knees, ankles...)"""
    cv2.circle(
        frame,
        (
            int(shot_inst[f"{key_point_str}_x"] * frame.shape[1]),
            int(shot_inst[f"{key_point_str}_y"] * frame.shape[0]),
        ),
        radius=3,
        color=(0, 255, 0),
        thickness=-1,
    )


def draw_edge(frame, shot_inst, edge):
    """Draw edges corresponding to members"""
    first_point = [
        keypoint for keypoint, value in KEYPOINT_DICT.items() if value == edge[0][0]
    ]
    if len(first_point) == 0:
        return
    first_point = first_point[0]

    second_point = [
        keypoint for keypoint, value in KEYPOINT_DICT.items() if value == edge[0][1]
    ]
    if len(second_point) == 0:
        return
    second_point = second_point[0]

    cv2.line(
        frame,
        (
            int(shot_inst[f"{first_point}_x"] * WIDTH),
            int(shot_inst[f"{first_point}_y"] * HEIGHT),
        ),
        (
            int(shot_inst[f"{second_point}_x"] * WIDTH),
            int(shot_inst[f"{second_point}_y"] * HEIGHT),
        ),
        color=COLORS[edge[1]],
        thickness=2,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize human poses from a csv file")
    parser.add_argument("shots", nargs="*", help="csv file(s)")
    parser.add_argument("--gif", type=str, help="Export shot as a gif")
    args = parser.parse_args()

    for shot_path in args.shots:
        shot = pd.read_csv(shot_path)
        shot = shot.loc[:, shot.columns != "shot"]
        if args.gif:
            frames = frames = np.zeros((len(shot), HEIGHT, WIDTH, 3), np.uint8)

        for i in range(len(shot)):
            frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            shot_inst = shot.iloc[i, :]
            for key_point in KEYPOINT_DICT:
                draw_key_point(frame, shot_inst, key_point)

            for edge in EDGES.items():
                draw_edge(frame, shot_inst, edge)

            cv2.putText(
                frame,
                shot_path,
                (10, WIDTH - 10),
                fontScale=1,
                color=(255, 255, 255),
                thickness=1,
                fontFace=1,
            )

            if args.gif:
                frames[i] = frame

            cv2.imshow("Shot", frame)
            k = cv2.waitKey(0)

            if k == 27:
                cv2.destroyAllWindows()
                break

        if args.gif:
            imageio.mimsave(args.gif, frames, fps=30)
