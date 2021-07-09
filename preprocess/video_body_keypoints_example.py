import os
import sys
import cv2
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../openpose/build/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../openpose/build/x64/Release;' + dir_path + '/../openpose/build/bin;'
# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
'''try:
    # Windows Import
    if platform == "win32":
    # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
'''
# import openpose_python as op
import pyopenpose as op

params = dict()
# params['model_folder'] = '../openpose/models'
params['model_folder'] = '../models'

params['model_pose'] = 'BODY_25'
params['number_people_max'] = 1

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()

point_pairs = [(0, 1), (2, 1), (2, 3), (3, 4), (5, 1), (5, 6), (6, 7), (8, 1), (8, 9), (9, 10), (10, 11),
               (8, 12), (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18), (14, 19), (14, 21), (19, 20),
               (11, 22), (23, 22), (11, 24)]

cap = cv2.VideoCapture("mod_code/openpose/code/video/normal/adl-01-cam0.mp4")
FPS = cap.get(cv2.CAP_PROP_FPS)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output.mp4', fourcc, FPS, (int(W), int(H)))

while cv2.waitKey(10) & 0xFF != ord('q'):
    success, frame = cap.read()

    if not success:
        break

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    keypoints_img = np.zeros_like(frame)

    if len(datum.poseKeypoints.shape) > 0:

        keypoints = datum.poseKeypoints[0]
        for pair in point_pairs:
            p1, p2 = pair

            if keypoints[p1][2] > 0 and keypoints[p2][2] > 0:
                cv2.line(keypoints_img, (keypoints[p1][0], keypoints[p1][1]),
                         (keypoints[p2][0], keypoints[p2][1]), (0, 255, 255), 5)

        for keypoint in datum.poseKeypoints[0]:
            if keypoint[2] > 0:
                cv2.circle(keypoints_img, (keypoint[0], keypoint[1]), 10, (0, 0, 255), -1)

    concat = cv2.hconcat([datum.cvOutputData, keypoints_img])
    cv2.imshow('show', concat)