from typing import Union, Callable

import cv2
import numpy as np

import pyopenpose as op

from cv2_utils import get_frame_from_cap

__all__ = ["HeatmapDisplay"]


def main():
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict(
        model_folder='models',
        heatmaps_add_bkg=True,
        heatmaps_add_PAFs=True,
        heatmaps_scale=2
    )

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
    heatmap_display = HeatmapDisplay(0.5, 0.5, 64)
    heatmap_display.initialize_tracerbars("frame")

    out_video = cv2.VideoWriter('heatmap_output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 7.0, (992, 368))
    for frame_num, frame in enumerate(get_frame_from_cap(cv2.VideoCapture(0))):
        # Process Image
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Process outputs
        heatmaps = datum.poseHeatMaps.copy()
        combined = heatmap_display.get_image(frame, heatmaps)

        cv2.imshow("frame", combined)
        out_video.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    out_video.release()


def minmax_scale(arr: np.ndarray, to_min: float, to_max: float, axis: Union[tuple, int] = None) -> np.ndarray:
    shape = list(arr.shape)
    if isinstance(axis, int):
        axis = (axis,)
    is_tuple = isinstance(axis, tuple)
    if not (axis is None or is_tuple):
        raise TypeError("argument axis should be a int, tuple, or None.")
    if axis is None:
        axis = tuple(range(len(shape)))
    for i in axis:
        shape[i] = 1
    arr_min = np.min(arr, axis=axis).reshape(shape)
    arr_max = np.max(arr, axis=axis).reshape(shape)
    return (arr - arr_min) / (arr_max - arr_min) * (to_max - to_min) + to_min


class HeatmapDisplay:
    original_heatmap_alpha: float
    part_pafs_alpha: float

    max_level: int

    TRACER_NAMES = ("original_heatmap_alpha", "part_pafs_alpha")

    def __init__(self, original_heatmap_alpha: float, part_pafs_alpha: float, max_level: int):
        self.original_heatmap_alpha = original_heatmap_alpha
        self.part_pafs_alpha = part_pafs_alpha
        self.max_level = max_level

    def initialize_tracerbars(self, winname: str):
        for name in self.TRACER_NAMES:
            cv2.createTrackbar(
                name,
                winname,
                int(getattr(self, name) * self.max_level),
                self.max_level,
                self.get_param_updater(name)
            )

    def get_image(self, frame: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
        bkg_inv = 255 - np.uint8(heatmaps[0])
        heatmaps = heatmaps[1:]
        heatmap = np.uint8(minmax_scale(np.sum(np.logical_and(heatmaps != 128, heatmaps != 129), axis=0), 0, 255))
        heatmap = cv2.addWeighted(bkg_inv, self.part_pafs_alpha, heatmap, 1 - self.part_pafs_alpha, 1)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_h, heatmap_w = heatmap.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        adjusted_heatmap_size = int(heatmap_w * frame_h / heatmap_h), frame_h
        heatmap = cv2.resize(heatmap, adjusted_heatmap_size)
        if heatmap.shape != frame.shape:
            heatmap = heatmap[:frame_h, :frame_w]
        return cv2.addWeighted(frame, self.original_heatmap_alpha, heatmap, 1 - self.original_heatmap_alpha, 1)

    def get_param_updater(self, name: str) -> Callable[[int], None]:
        def _updater(val: int):
            setattr(self, name, val / self.max_level)

        return _updater


if __name__ == '__main__':
    main()
