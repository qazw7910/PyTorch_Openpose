import env_path
import cv2
import numpy as np
import argparse
import torch
from keypoint_models.models import Keypoint_LSTM

from preprocess.BODY_25 import BODY_25
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1394
import pyopenpose as op
from imutils.video import FPS
import time


def fill_null(features):
    seq_length = len(features)
    new_features = np.array(features)

    for t in range(seq_length):
        for point in selected_points:
            if new_features[t][point.value][2] == 0:
                lt, rt = t, t
                while lt >= 0 and new_features[lt][point.value][2] == 0:
                    lt -= 1
                while rt < seq_length and new_features[rt][point.value][2] == 0:
                    rt += 1

                value = [0, 0]
                new_features[t, point.value, 2] = 0.5

                if lt < 0 and rt < seq_length:
                    value = new_features[rt, point.value, :2]
                elif rt >= seq_length and lt >= 0:
                    value = new_features[lt, point.value, :2]
                elif lt >= 0 and rt < seq_length:
                    diff_vec = new_features[rt, point.value, :2] - new_features[lt, point.value, :2]
                    value = new_features[lt, point.value, :2] + diff_vec / (rt - lt) * (t - lt)
                else:
                    new_features[t, point.value, 2] = 0

                new_features[t, point.value, :2] = value

    return new_features

def normalization(features, shape):
    norm_features = np.array(features)

    norm_features[:, :, 0] = norm_features[:, :, 0] / shape[0]
    norm_features[:, :, 1] = norm_features[:, :, 1] / shape[1]

    return norm_features

class main :

    def __init__(self, source):
        self.op_params = self.set_op_params()
        self.stream = cv2.VideoCapture(source)
        self.num_gpus = self.op_params['num_gpu'] if'num_gpu' in self.op_params else op.get_gpu_number()

    def set_op_params(self):
        return dict(
            model_folder='bin/models',
            model_pose='BODY_25',
            frame_step=2,
            process_real_time='true',
            render_threshold=0.5
        )

    def start(self):
        buffer = list()
        fps = FPS().start()
        exit = False

        start =time.time()


        op_wrapper = op.WrapperPython()
        op_wrapper.configure(self.op_params)
        op_wrapper.start()

        model_time_step = 20
        real_time_step = 0
        num_keypoint = 25
        feature_array = np.zeros([45, num_keypoint, 3], np.float32)
        class_name = ['fall', 'stand']
        state = 'unknown'
        choosed_confidence = 0
        while self.stream.isOpened():
            sucess, frame = self.stream.read()
            fps = self.stream.get(cv2.CAP_PROP_FPS)

            if frame is None or not sucess:
                print("Error while open capture device")
                break

            buffer.append(frame)

            if len(buffer) >= self.num_gpus:

                for image_base_id in range(0, len(buffer), self.num_gpus):
                    datums = []
                    video_info = {}
                    W = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                    H = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    video_info['W'] = int(W)
                    video_info['H'] = int(H)



                    # catch keypoints and recognize state
                    for gpuId in range(0, self.num_gpus):
                        if real_time_step < model_time_step:
                            datum = op.Datum()
                            datum.cvInputData = frame
                            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                            # add keypoint to feature_array[frame_cnt, :, :]
                            keypoints = datum.poseKeypoints[0]
                            feature_array[real_time_step, :, :] = keypoints
                            real_time_step += 1

                        else:
                            # delivery norm_keypoints_buffer to model recognize and return body state
                            fill_features = fill_null(feature_array)
                            norm_features = normalization(fill_features, (video_info['W'], video_info['H']))

                            # filter norm_features keypoint and reshape
                            seq_length = len(norm_features)
                            filter_norm_features = norm_features[:, selected_points_index,:2]
                            filter_norm_features = filter_norm_features.reshape((seq_length, len(selected_points)*2))

                            # convert filter_norm_features to filter_norm_features_tensor also put in cuda
                            filter_norm_features_tensor = torch.from_numpy(filter_norm_features).type(torch.float32)
                            X = filter_norm_features_tensor.cuda().reshape(1,*filter_norm_features_tensor.size())

                            # load model and convert model to eval() mode
                            model = Keypoint_LSTM(input_size=30, hidden_size=64, num_layers=1, num_classes=2)
                            model.load_state_dict(torch.load(f'pt_model/fall_{model_time_step}_fps.pth'))
                            model.eval().cuda()

                            # model output state
                            output = model(X)
                            confidence = torch.softmax(output, dim=1)
                            idx = confidence.argmax(dim=1)[0].item()
                            choosed_confidence = confidence[0,idx]
                            if choosed_confidence >= confidence_threshold:
                                state = class_name[idx]
                            else:
                                state = 'unknown'

                            # after delivery feature_array need to clear
                            feature_array = np.zeros([45, num_keypoint, 3], np.float32)
                            real_time_step = 0

                    #Read and push images into OpenPose wrapper
                    for gpuId in range(0, self.num_gpus):
                        image_id = image_base_id + gpuId

                        if image_id < len(buffer):
                            frame = buffer[image_id]
                            datum = op.Datum()
                            datum.cvInputData = frame
                            datums.append(datum)
                            op_wrapper.waitAndEmplace(op.VectorDatum([datums[-1]]))

                    # Retrieve processed results from OpenPose wrapper
                    for gpuId in range(0, self.num_gpus):
                        image_id = image_id + gpuId

                        if image_id < len(buffer):
                            datum = datums[gpuId]
                            op_wrapper.waitAndPop(op.VectorDatum([datum]))

                            op_frame = datum.cvOutputData

                            fps.update()
                            fps.stop()

                            cv2.putText(img=op_frame, text='NUM_GPU: {}'.format(self.num_gpus), org=(30,50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
                            cv2.putText(img=op_frame, text='FPS(): {0:.2f}'.format(fps.fps()), org=(30,90),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
                            #TODO STATE = after model recognize
                            cv2.putText(img=op_frame, text=f'STATE: {state}' ,org=(30,130),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
                            cv2.putText(img=op_frame, text=f'CONF: {choosed_confidence: .6f}', org=(30, 160),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                                        thickness=2)

                            op_frame = cv2.resize(src=op_frame, dsize=(0,0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                            cv2.imshow('fall recognizer', op_frame)

                            key = cv2.waitKey(1) & 0xFF
                            if key ==ord('q') or key == 27:
                                print('closing program')
                                exit = True
                                break

                    if exit:
                        break

                buffer.clear()

            if exit:
                break
            cv2.waitKey(0)
            break

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
        self.stream.release()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    parse =argparse.ArgumentParser()
    parse.add_argument('--source', required=True, help='source need to process')
    args =vars(parse.parse_args())

    selected_points = [BODY_25.Nose, BODY_25.Neck, BODY_25.RShoulder, BODY_25.RElbow, BODY_25.RWrist,
                       BODY_25.LShoulder, BODY_25.LElbow, BODY_25.LWrist, BODY_25.MidHip,
                       BODY_25.RHip, BODY_25.RKnee, BODY_25.RAnkle, BODY_25.LHip, BODY_25.LKnee,
                       BODY_25.LAnkle]

    selected_points_index = [point.value for point in selected_points]

    confidence_threshold = 0.6

    print(selected_points_index)

    source = args['source']

    main = main(source = source)
    main.start()