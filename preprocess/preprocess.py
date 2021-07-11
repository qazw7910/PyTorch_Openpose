import subprocess
import os
import cv2
import re
import numpy as np

# https: // www.runoob.com / w3cnote / python3 - subprocess.html
def check_rotation(video_path):
    rotation = None

    p = subprocess.Popen('ffprobe -v error -show_format -show_streams ' + video_path, shell = True, stdout=subprocess.PIPE, stderr = subprocess.PIPE)

    out, err = p.communicate()

    out = out.decode('BIG5')
    err = err.decode('BIG5')
    
    if p.returncode != 0:
        print(err)
        return None
    
    res = re.search(r'TAG:rotate=\d+', out)
    
    if res is not None:
        res = res.group()
        rotation = int(res.split('=')[-1])

    return rotation


def correct_video_rotation(video_path, save_path):
    
    rotation = check_rotation(video_path)
    
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if rotation == 90 or rotation == 270:
        vout = cv2.VideoWriter(save_path, fourcc, fps, (height, width))
    else:
        vout = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for _ in range(framecnt):

        ret, frame = cap.read()
        
        if rotation is not None:
            frame = np.rot90(frame, (360 - rotation) // 90)

        vout.write(frame)

    cap.release()
    vout.release()


def extract_rgb(video_path, save_dir, newsize = (0, 0)):

    cap = cv2.VideoCapture(video_path)
    framecnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # create save directory if not exist
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    
    for i in range(1, framecnt+1):
        ret, frame = cap.read()

        # resize frame if given newsize
        if newsize != (0, 0):
            
            if isinstance(newsize, tuple):
                frame = cv2.resize(frame, newsize)
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if width >= height:
                    new_width = int(width * newsize / height)
                    new_height = newsize
                else:
                    new_width = newsize
                    new_height = int(height * newsize / width)
                    
                newsize = (new_width, new_height)
                frame = cv2.resize(frame, newsize)

        cv2.imwrite(os.path.join(save_dir, f'rgb_{i:04d}.jpg'), frame)
    
    cap.release()


def build_info_file(data_root, save_dir):
    with open(os.path.join(save_dir, 'info.txt'), 'w') as infofile:

        cnt = 1
        
        for r, d, files in os.walk(data_root):
            for f in files:
                out = f'id={cnt},'
                cnt += 1

                path = os.path.join(r, f)
                out += f'path={path},'

                cap = cv2.VideoCapture(path)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                out += f'num_frames={num_frames},'

                if 'fall' in f:
                    label = 'fall'
                elif 'stand' in f:
                    label = 'stand'
                '''elif 'sit' in f:
                    label = 'sit'
                elif 'crouch' in f:
                    label = 'crouch'
'''
                out += f'label={label}\n'

                infofile.write(out)


def read_info_file(info_file_path):
    with open(info_file_path, 'r') as f:
        info = f.read()
    info = info.split()
    
    data_list = []
    for data in info:
        data = data.split(',')
        m = {}
        for item in data:
            name, value = item.split('=')

            try:
                value = int(value)
            except ValueError:
                pass

            m[name] = value
        data_list.append(m)
    return data_list


if __name__ == '__main__':

    #build_info_file('video','extract')
    read_info_file('extract/info.txt')