import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import copy

from model import TSN
from transforms import *
from dataset import TSNDataSet


def main():
    info_list = read_info_file('extract/info.txt')

    batch_size = 32
    num_segments = 10
    new_length = 5
    modality = 'RGBdiff'
    num_epochs = 150

    train_sample_size = int(len(info_list) * 0.5)

    np.random.seed(15)
    randidx = np.random.permutation(len(info_list))

    train_info_list = [info_list[idx] for idx in randidx[:train_sample_size]]

    train_dataset = TSNDataSet(info_list=train_info_list,
                               data_root='mod_code/openpose/code/extract/info.txt',
                               num_segments=num_segments,
                               new_length=new_length,
                               modality=modality,
                               transform=transforms.Compose([
                                   GroupScaleJitterCrop([1.0, 0.9, 0.8], [1.0, 0.9, 0.8]),
                                   GroupResize((224, 224)),
                                   GroupRandomHorizontalFlip(),
                                   GroupToTensor(),
                                   Stack()
                               ]))

    weight = train_dataset.get_data_weight()

    sampler = torch.utils.data.WeightedRandomSampler(weights=weight,
                                                     num_samples=train_sample_size,
                                                     replacement=True)

    val_info_list = [info_list[idx] for idx in randidx[train_sample_size:]]

    model = TSN(num_class=4, num_segments=num_segments, new_length=new_length,
                modality=modality, base_model='resnet18', dropout=0)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    mean = model.input_mean
    std = model.input_std

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=sampler,
                              num_workers=3)

    val_loader = DataLoader(TSNDataSet(info_list=val_info_list,
                                       data_root='extract\\',
                                       num_segments=num_segments,
                                       new_length=new_length,
                                       modality=modality,
                                       transform=transforms.Compose([
                                           GroupResize((224, 224)),
                                           GroupToTensor(),
                                           Stack()
                                       ]),
                                       random_sample=False),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=3)

    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    dataloaders = {'train': train_loader, 'val': val_loader}

    train_res = train_model(model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                            num_epochs=num_epochs, device=device)

    model, train_acc_history, train_loss_history, val_acc_history, val_loss_history = train_res

    torch.save(model.state_dict(), 'mod_code/openpose/models\\rgb_60seg.pth')

    plot_and_save(range(num_epochs), [train_loss_history, val_loss_history],
                  '# of iterations', 'loss', 'figure\\rgb_60seg_loss.jpg',
                  legends=['train', 'validation'])

    plot_and_save(range(num_epochs), [train_acc_history, val_acc_history],
                  '# of iterations', 'accuracy', 'figure\\rgb_60seg_acc.jpg',
                  legends=['train', 'validation'])


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc * 100)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc * 100)
                train_loss_history.append(epoch_loss)

        if epoch == 49:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history


def plot_and_save(x_val, y_val, x_label, y_label, save_path, legends=None):
    plt.clf()

    for y in y_val:
        plt.plot(x_val, y, '-')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    x_ticks = list(range(0, len(x_val), 50))
    plt.xticks(x_ticks)

    if legends:
        plt.legend(legends)

    plt.savefig(save_path)


if __name__ == '__main__':
    #main()
    train_model()

