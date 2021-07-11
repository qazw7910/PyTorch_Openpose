import torch
import torch.nn as nn


class Keypoint_LSTM(nn.Module):
    # TODO num_classes=4
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2, batch_first=True):
        super(Keypoint_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, dropout=0.3)
        self.drop = nn.Dropout(0.)
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, packed_input):
        '''
        h0 = torch.rand(self.hidden_size, self.num_layers)
        c0 = torch.rand(self.hidden_size, self.num_layers)
        '''
        packed_output, (ht, ct) = self.lstm(packed_input)

        output = self.drop(ht[-1])
        output = self.nn(output)

        return output


# https://blog.csdn.net/sunny_xsc1994/article/details/82969867

class Conv1D(nn.Module):
    def __init__(self, signal_length: int, num_classes: int):
        super(Conv1D, self).__init__()

        self.num_kernel = 64
        self.signal_length = signal_length
        self.num_classes = num_classes
        if signal_length // 45 ==0:
            raise ValueError(f'signal_length too small: {signal_length}')

        self.conv = nn.Sequential(
            nn.Conv1d(30, self.num_kernel, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(5),
            nn.Conv1d(self.num_kernel, self.num_kernel, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(self.num_kernel, self.num_kernel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(self.num_kernel, self.num_kernel, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(self.num_kernel * (self.signal_length // 45), 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, in_x):
        out = self.conv(in_x)
        out = self.fc(out.reshape(in_x.size(0), -1))
        return out
