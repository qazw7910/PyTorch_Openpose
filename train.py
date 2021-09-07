import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from keypoint_models.models import *
from keypoint_models.dataset import BodyKeypointsDataset
from keypoint_models.transform import *
from preprocess.BODY_25 import BODY_25


def train_model(model, train_loader, val_loader, test_loader, epoch):  #test loader
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    loss_history = {'train': [], 'val': [], 'test': []}
    acc_history = {'train': [], 'val': [], 'test': []}
    dataloaders = {'train': train_loader, 'val': val_loader , 'test': test_loader}
    best_val_acc = 0

    for i in range(epoch):
        if i == 500:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        elif i == 800:
            optimizer = optim.Adam(model.parameters(), lr=0.00005)

        for phase in ['train', 'val', 'test']:
            epoch_loss = 0.0
            epoch_correct = 0
            num_data = 0

            dataset.set_phase(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in dataloaders[phase]:
                X = batch['feature']
                y = batch['label']

                X = X.cuda()
                y = y.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    output = model(X)
                    loss = criterion(output, y)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                        optimizer.step()

                pred = output.argmax(dim=1)

                num_data += len(y)
                epoch_correct += (pred == y).sum().item()
                epoch_loss += loss.item() * len(y)

            loss = epoch_loss / num_data
            acc = epoch_correct / num_data *100
            loss_history[phase].append(loss)
            acc_history[phase].append(acc)

            if phase in ['val', 'test']:
                best_val_acc = max(best_val_acc, acc)

    for phase in ['train', 'val', 'test']:
        plt.plot(range(len(loss_history[phase])), loss_history[phase])
    plt.legend(['train', 'validation', 'test'])
    plt.xlabel('# of iterations')
    plt.ylabel('loss')
    plt.show()

    plt.clf()
    for phase in ['train', 'val', 'test']:
        plt.plot(range(len(acc_history[phase])), acc_history[phase])
    plt.legend(['train', 'validation', 'test'])
    plt.xlabel('# of iterations')
    plt.ylabel('accuracy')
    plt.show()

    return best_val_acc


keypoints = [BODY_25.Nose, BODY_25.Neck, BODY_25.RShoulder, BODY_25.RElbow, BODY_25.RWrist,
             BODY_25.LShoulder, BODY_25.LElbow, BODY_25.LWrist, BODY_25.MidHip,
             BODY_25.RHip, BODY_25.RKnee, BODY_25.RAnkle, BODY_25.LHip, BODY_25.LKnee,
             BODY_25.LAnkle]
keypoints = [point.value for point in keypoints]

num_feature = 2 * len(keypoints)  # because(X,Y) so feature *2

torch.manual_seed(0)

train_transforms = transforms.Compose([Centralize(),
                                       Scale(0.7, 1.3),
                                       RandomShift(-0.3, 0.3, -0.3, 0.3)
                                       ]) #for lstm

val_transforms = transforms.Compose([])

test_transforms = transforms.Compose([])

time_steps = 45
dataset = BodyKeypointsDataset(keypoints, root_dir='data', timesteps=time_steps, transforms={'train': train_transforms, 'val': val_transforms, 'test':test_transforms},
                               pad_by_last=True)

# split_lengths = [len(dataset) // 2, len(dataset) - len(dataset) // 2]
train_size =int(len(dataset) * 0.5)
val_size =int(len(dataset) * 0.3)
test_size =int(len(dataset)) - train_size - val_size
train_dataset, val_dataset ,test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


cnt_class = [0, 0]    #TODO cnt_class = [0, 0, 0, 0]
num_classes = len(cnt_class)
for data in train_dataset:
    cnt_class[data['label']] += 1

class_weight = [1 / cnt for cnt in cnt_class]

train_size = len(train_dataset)
data_weight = [0] * train_size
for i, data in enumerate(train_dataset):
    label = data['label']
    data_weight[i] = class_weight[label]

sampler = torch.utils.data.WeightedRandomSampler(weights=data_weight, num_samples=512, replacement=True)    #https://blog.csdn.net/tyfwin/article/details/108435756


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)


# model = Conv1D(signal_length=time_steps, num_classes=num_classes)
model = Keypoint_LSTM(
    input_size=30,
    hidden_size=64,
    num_layers=1,
    num_classes=2
)
res = train_model(model, train_loader, val_loader, test_loader, 18)
print('best_val_acc:', res)

torch.save(model.state_dict(), 'pt_model/fall.pth')

'''model = Conv1D(signal_length=time_steps, num_classes=num_classes)
model.load_state_dict(torch.load('pt_model/fall.pth'))
model.eval()'''