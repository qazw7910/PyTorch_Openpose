import os
import sys
import cv2
import numpy as np
import argparse

from BODY_25 import BODY_25


import pyopenpose as op


def command_parser():
    parser = argparse.ArgumentParser(description='program useage description')

    parser.add_argument('--save_dir', type=str,
                        help='Save extract keypoints .npy files to this directory.')
    parser.add_argument('--video_dir', type=str,
                        help='Extract human body keypoints from videos in this directory.')
    parser.add_argument('--video_path', type=str,
                        help='Extract human body keypoints from given single video.')
    parser.add_argument('--truncate', type=float, default=0, help='Truncate last n seconds of video.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit()

    return parser.parse_args()

# 確認檔案格式
def is_video(filename):
    filename = filename.lower()
    videofile_extensions = ['.mp4', '.avi', '.mpeg', ]

    for extension in videofile_extensions:
        if filename.endswith(extension):
            return True
    return False

# 大於threshold值
def is_confidential_candidate(candidate, sel_points, threshold=0.0):
    for point in sel_points:
        confident = candidate[point.value][2]
        if 0 < confident < threshold:
            return False

    return True


def is_good_video(features):
    seq_length = len(features)
    num_points = 25
    points_cnt = np.zeros(num_points)

    # Iterate each time step in feature
    for ti in features:
        # Iterate each selected point in each time step, and count how many accepted points there are in video
        for point in selected_points:
            if ti[point.value][2] > 0:
                points_cnt[point.value] += 1

    # TODO write condition to define what is a good video.
    for point in selected_points:
        if points_cnt[point.value] < seq_length * 0.1:  # 0.8
            return False

    return True


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


def draw_body_points(key_points, img_shape, origin_img=None):
    if origin_img is not None:
        img = np.array(origin_img)
    else:
        img = np.zeros(img_shape, dtype=np.uint8)

    for pair in BODY_25.get_pairs():
        if key_points[pair[0]][2] > 0 and key_points[pair[1]][2] > 0:
            pt1 = (key_points[pair[0]][0], key_points[pair[0]][1])
            pt2 = (key_points[pair[1]][0], key_points[pair[1]][1])
            cv2.line(img, pt1, pt2, (0, 255, 255), 5)

    for key_point in key_points:
        if key_point[2] > 0:
            cv2.circle(img, (key_point[0], key_point[1]), 5, (0, 0, 255), -1)

    return img


def show_images(images_list):
    list_length = len(images_list)
    if list_length == 0:
        return
    elif list_length == 1:
        combine = images_list[0]
    elif list_length == 2:
        combine = cv2.hconcat(images_list)
    elif list_length == 4:
        combine = cv2.vconcat([cv2.hconcat(images_list[:2]), cv2.hconcat(images_list[2:])])

    cv2.imshow('Visualization', combine)

    comm = cv2.waitKey() & 0xFF#q離開

    if comm == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
    elif comm == ord('s'):
        cv2.imwrite('pic.jpg', combine)


def extract(video_path, show_video=False):
    cap = cv2.VideoCapture(video_path)

    video_info = {}
    FPS = cap.get(cv2.CAP_PROP_FPS)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FRAME_CNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    video_info['FPS'] = FPS
    video_info['W'] = int(W)
    video_info['H'] = int(H)
    video_info['FRAME_CNT'] = int(FRAME_CNT)

    seq_length = int(FRAME_CNT - args.truncate * FPS)

    # BODY_25 model extract 25 body keypoints
    num_keypoint = 25
    feature_array = np.zeros([seq_length, num_keypoint, 3], np.float32)
    frame_cnt = 0

    while frame_cnt < seq_length:
        success, frame = cap.read()
        if not success:
            print('retrieve frame from', file_path, 'failed.', file=sys.stderr)
            sys.exit(-1)    #停止繼續執行

        datum.cvInputData = frame
        op_wrapper.emplaceAndPop([datum])

        # sometimes model doesn't detect any person in a frame,
        # so copy keypoints to array only when it detects a person
        # and the keypoints we want is confidential enough
        if len(datum.poseKeypoints.shape) > 0 and is_confidential_candidate(datum.poseKeypoints[0], selected_points,
                                                                            threshold=0.3):
            # index represents n-th person
            keypoints = datum.poseKeypoints[0]
            feature_array[frame_cnt, :, :] = keypoints

        if show_video:
            body_points_img2 = draw_body_points(feature_array[frame_cnt, :, :], frame.shape)

            if len(datum.poseKeypoints.shape) > 0:
                body_points_img = draw_body_points(datum.poseKeypoints[0], frame.shape)
            else:
                body_points_img = body_points_img2

            show_images([body_points_img, body_points_img2])

        frame_cnt += 1

    return video_info, feature_array


def normalization(features, shape):
    norm_features = np.array(features)

    norm_features[:, :, 0] = norm_features[:, :, 0] / shape[0]
    norm_features[:, :, 1] = norm_features[:, :, 1] / shape[1]

    return norm_features


if __name__ == '__main__':
    args = command_parser()

    params = dict()
    params['model_folder'] = 'bin/models'
    params['model_pose'] = 'BODY_25'
    params['number_people_max'] = 1
#   https://yuanze.wang/posts/build-openpose-python-api/
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    datum = op.Datum()

    selected_points = [BODY_25.Nose, BODY_25.Neck, BODY_25.RShoulder, BODY_25.RElbow, BODY_25.RWrist,
             BODY_25.LShoulder, BODY_25.LElbow, BODY_25.LWrist,BODY_25.MidHip,
             BODY_25.RHip, BODY_25.RKnee, BODY_25.RAnkle, BODY_25.LHip, BODY_25.LKnee,
             BODY_25.LAnkle]

    if args.video_path:
        if is_video(args.video_path):
            video_info, features = extract(args.video_path)

            new_features = fill_null(features)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer1 = cv2.VideoWriter('origin.mp4', fourcc, video_info['FPS'], (int(video_info['W']), int(video_info['H'])))
            writer2 = cv2.VideoWriter('fill.mp4', fourcc, video_info['FPS'],
                                      (int(video_info['W']), int(video_info['H'])))

            cap = cv2.VideoCapture(args.video_path)
            for t in range(len(features)):
                success, frame = cap.read()

                video_origin_feature_img = draw_body_points(features[t, :, :], (video_info['H'], video_info['W'], 3), frame)
                video_new_feature_img = draw_body_points(new_features[t, :, :], (video_info['H'], video_info['W'], 3), frame)
                origin_feature_img = draw_body_points(features[t, :, :], (video_info['H'], video_info['W'], 3))
                new_feature_img = draw_body_points(new_features[t, :, :], (video_info['H'], video_info['W'], 3))

                '''
                writer1.write(video_origin_feature_img)
                writer2.write(video_new_feature_img)
                '''

                show_images([video_origin_feature_img, video_new_feature_img])

        else:
            print('Can\'t read video from the given path.\n' + '\''+args.video_path+'\'', file=sys.stderr)
            sys.exit(-1)

    elif args.video_dir:
        root, _, files = next(os.walk(args.video_dir))

        file_cnt = 0
        for file in files:
            file_path = os.path.join(root, file)
            if is_video(os.path.join(file_path)):
                video_info, features = extract(file_path)

                if is_good_video(features):
                    fill_features = fill_null(features)
                    norm_features = normalization(fill_features, (video_info['W'], video_info['H']))

                    save_path = os.path.join(args.save_dir, f'{file_cnt:04d}.npy')
                    np.save(save_path, norm_features)

                    file_cnt += 1
                    print('good video.')
                else:
                    print('bad video.')

                print(f'Video \'{file}\' done.')
