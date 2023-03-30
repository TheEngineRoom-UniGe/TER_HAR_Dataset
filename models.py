import torch.nn as nn


class LSTMMultiClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob=0.5):
        super(LSTMMultiClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, dropout=dropout_prob, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, output_dim)
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # out = self.dropout(lstm_out[:, -1, :])
        out = self.fc1(lstm_out[:, -1, :])
        # out = self.relu(out)
        out = self.fc2(out)
        out = self.sm(out)
        return out


class LSTMBinary(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob=0.5):
        super(LSTMBinary, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, dropout=dropout_prob, batch_first=True)
        # self.fc1 = nn.Linear(hidden_dim, 128)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(128, output_dim)
        # self.sm = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # out = self.dropout(lstm_out[:, -1, :])
        out = self.dropout(lstm_out[:, -1, :])
        # out = self.relu(out)
        out = self.fc(out)
        out = self.sig(out)
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_dim=128, n_layers=2, n_heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = n_features, nhead = n_heads),
            num_layers = n_layers
            )
        # self.fc1 = nn.Linear(n_features, hidden_dim)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x