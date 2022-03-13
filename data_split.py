import os
import random
import math
import shutil

# https://blog.csdn.net/sinat_35907936/article/details/105611737

def data_split(old_data):
    new_path = 'data_classification'
    if os.path.exists('data_classification') == 0:
        os.mkdir(new_path)

    for root_dir, sub_dirs, file in os.walk(old_data):
        for sub_dir in sub_dirs:
            file_name = os.listdir(os.path.join(root_dir, sub_dir))
            file_name = list(filter(lambda x: x.endswith('.npy'), file_name))

            random.shuffle(file_name)

            for i in range(len(file_name)):
                if i < math.floor(0.8 * len(file_name)):
                    sub_path = os.path.join(new_path, train_dataset, sub_dir)
                    if os.path.exists(f'{new_path}\{train_dataset}') == 0:
                        os.mkdir(f'{new_path}\{train_dataset}')
                elif i < math.floor(0.9 * len(file_name)):
                    sub_path = os.path.join(new_path, val_dataset, sub_dir)
                    if os.path.exists(f'{new_path}\{val_dataset}') == 0:
                        os.mkdir(f'{new_path}\{val_dataset}')
                elif i < len(file_name):
                    sub_path = os.path.join(new_path, test_dataset, sub_dir)
                    if os.path.exists(f'{new_path}\{test_dataset}') == 0:
                        os.mkdir(f'{new_path}\{test_dataset}')

                if os.path.exists(sub_path) == 0:
                    os.mkdir(sub_path)

                shutil.copy(os.path.join(root_dir, sub_dir, file_name[i]), os.path.join(sub_path, file_name[i]))



if __name__ == '__main__':

    train_dataset = 'train_dataset'
    val_dataset = 'val_dataset'
    test_dataset = 'test_dataset'

    data_path = 'data'
    data_split(data_path)
