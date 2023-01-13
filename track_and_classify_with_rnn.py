"""
With this script, you can provide a video and your RNN model (e.g tennis_rnn.h5)
and see a shot classification/detection.For this, we feed our neural network with
a sliding window of 30 frame (1 second) and classify the shot.
Same kind of shot counter is used then.
"""

import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2

from extract_human_pose import HumanPoseExtractor

physical_devices = tf.config.experimental.list_physical_devices("GPU")
print(tf.config.experimental.list_physical_devices("GPU"))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


class ShotCounter:
    """
    Pretty much the same principle than in track_and_classify_frame_by_frame
    except that we dont have any history here, and confidence threshold can be much higher.
    """

    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []

    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""

        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs

        if (
            probs[0] > 0.9
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            probs[1] > 0.9
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            len(probs) > 3
            and probs[3] > 0.5
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1

    def display(self, frame):
        """Display counter"""
        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "backhand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "forehand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "serve" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )


BAR_WIDTH = 30
BAR_HEIGHT = 170
MARGIN_ABOVE_BAR = 30
SPACE_BETWEEN_BARS = 55
TEXT_ORIGIN_X = 1075
BAR_ORIGIN_X = 1070


def draw_probs(frame, probs):
    """Draw vertical bars representing probabilities"""

    cv2.putText(
        frame,
        "S",
        (TEXT_ORIGIN_X, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "B",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "N",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 2, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "F",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 3, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[3]),
        ),
        (BAR_ORIGIN_X + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
        color=(0, 0, 255),
        thickness=-1,
    )

    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[0]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[2]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[1]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    for i in range(4):
        cv2.rectangle(
            frame,
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i,
                int(MARGIN_ABOVE_BAR),
            ),
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i + BAR_WIDTH,
                BAR_HEIGHT + MARGIN_ABOVE_BAR,
            ),
            color=(255, 255, 255),
            thickness=1,
        )

    return frame


class GT:
    """GT to optionnally assess your results"""

    def __init__(self, path_to_annotation):
        self.shots = pd.read_csv(path_to_annotation)
        self.current_row_in_shots = 0
        self.nb_backhands = 0
        self.nb_forehands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"

    def display(self, frame, frame_id):
        """Display shot counter"""
        if self.current_row_in_shots < len(self.shots):
            if frame_id == self.shots.iloc[self.current_row_in_shots]["FrameId"]:
                if self.shots.iloc[self.current_row_in_shots]["Shot"] == "backhand":
                    self.nb_backhands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "forehand":
                    self.nb_forehands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "serve":
                    self.nb_serves += 1
                self.last_shot = self.shots.iloc[self.current_row_in_shots]["Shot"]
                self.current_row_in_shots += 1

        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (frame.shape[1] - 300, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "backhand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (frame.shape[1] - 300, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "forehand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (frame.shape[1] - 300, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "serve" else (0, 255, 0),
            thickness=2,
        )


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Used for debugging purpose"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def compute_recall_precision(gt, shots):
    """
    Assess your results against a Groundtruth
    like number of misses (recall) and number of false positives (precision)
    """

    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1

    precision = nb_match / (nb_match + nb_fp)
    recall = nb_match / (nb_match + nb_misses)

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(
        f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, serves = {fp_serves}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Track tennis player and display shot probabilities"
    )
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-f", type=int, help="Forward to")
    parser.add_argument(
        "--left-handed",
        action="store_const",
        const=True,
        default=False,
        help="If player is left-handed",
    )
    args = parser.parse_args()

    shot_counter = ShotCounter()

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.video)

    assert cap.isOpened()

    ret, frame = cap.read()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30

    FRAME_ID = 0

    features_pool = []

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_ID += 1

        if args.f is not None and FRAME_ID < args.f:
            continue

        assert frame is not None

        human_pose_extractor.extract(frame)

        # if not human_pose_extractor.roi.valid:
        #    features_pool = []
        #    continue

        # dont draw non-significant points/edges by setting probability to 0
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if args.left_handed:
            features[:, 1] = 1 - features[:, 1]

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

        features_pool.append(features)
        # print(features_pool)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = m1.__call__(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)

            # Give space to pool
            features_pool = features_pool[1:]

        draw_probs(frame, shot_counter.probs)
        shot_counter.display(frame)

        if args.evaluate is not None:
            gt.display(frame, FRAME_ID)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        draw_fps(frame, fps)
        draw_frame_id(frame, FRAME_ID)

        # Display results on original frame
        human_pose_extractor.draw_results_frame(frame)
        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        # cv2.imwrite(f"demo/image_{frame_id:04d}.png", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(shot_counter.results)

    if args.evaluate is not None:
        compute_recall_precision(gt.shots, shot_counter.results)
