from pathlib import Path
from typing import Iterator

import cv2
import flirpy.camera.core as core
from flirpy.camera.boson import Boson


def get_frame_from_cap(cap: cv2.VideoCapture) -> Iterator:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def get_frame_from_cam(camera: core.Core):
    with camera:
        while True:
            yield camera.grab()


def main():

    for frame in get_frame_from_cap(cv2.VideoCapture(0)):
        cv2.imshow("frame", frame)
        cv2.waitKey(1000//30)


if __name__ == '__main__':
    main()

