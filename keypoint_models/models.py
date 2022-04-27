import torch.nn as nn


class KeypointRNNBased(nn.Module):
    lstm: nn.RNNBase

    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2, batch_first=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = self._get_rnn_based_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.drop = nn.Dropout(0.3)
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, num_classes)

        )

    def forward(self, packed_input):
        """
        h0 = torch.rand(self.hidden_size, self.num_layers)
        c0 = torch.rand(self.hidden_size, self.num_layers)
        """
        packed_output, _ = self.lstm(packed_input)

        output = self.drop(packed_output[:, -1] if self.lstm.batch_first else packed_output[-1])
        output = self.nn(output)

        return output

    def _get_rnn_based_layer(self, input_size, hidden_size, num_layers, batch_first) -> nn.RNNBase:
        raise NotImplementedError()


class KeypointLSTM(KeypointRNNBased):
    def _get_rnn_based_layer(self, input_size, hidden_size, num_layers, batch_first) -> nn.RNNBase:
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )


class KeypointGRU(KeypointRNNBased):
    def _get_rnn_based_layer(self, input_size, hidden_size, num_layers, batch_first) -> nn.RNNBase:
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )


class Conv1D(nn.Module):
    def __init__(self, signal_length: int, num_classes: int):
        super(Conv1D, self).__init__()

        self.num_kernel = 64
        self.signal_length = signal_length
        self.num_classes = num_classes
        if signal_length // 45 == 0:
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
