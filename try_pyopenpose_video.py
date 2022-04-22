import time

import cv2
import pyopenpose as op

from cv2_utils import get_frame_from_cap

opWrapper = op.WrapperPython()
params = dict(
            model_folder='models',
            model_pose='BODY_25',
            net_resolution="320x160",
            frame_step=2,
            process_real_time='true',
            render_threshold=0.5
        )
print(params)
opWrapper.configure(params)
opWrapper.start()

# Process Image
cap = cv2.VideoCapture(0)

for frame in get_frame_from_cap(cap):
    datum = op.Datum()
    datum.cvInputData = frame

    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    cv2.imshow("frame", datum.cvOutputData)
    cv2.waitKey(1)

print("OpenPose demo successfully finished.")
