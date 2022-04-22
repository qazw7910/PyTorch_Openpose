from pathlib import Path
from typing import Iterator

import cv2


def get_frame_from_cap(cap: cv2.VideoCapture) -> Iterator:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def main():
    source = Path("video/IMG_3215.mp4")
    cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("frame", 1600, 900)

    for frame in get_frame_from_cap(cv2.VideoCapture(str(source))):
        cv2.imshow("frame", frame)
        cv2.waitKey(1000//30)


if __name__ == '__main__':
    main()

