import cv2
import pyopenpose as op

from try_opencv_stream import get_frame_from_cap

opWrapper = op.WrapperPython()
opWrapper.configure(dict(
            model_folder='models',
            model_pose='BODY_25',
            net_resolution="320x160",
            frame_step=2,
            process_real_time='true',
            render_threshold=0.5
        ))
opWrapper.start()

# Process Image
datum = op.Datum()
cap = cv2.VideoCapture("video/IMG_3215.mp4")

for frame in get_frame_from_cap(cap):
    # print(img.shape)
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (w//3, h//3))
    # print("image size after shrinking", img.shape)

    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(1000//30)
