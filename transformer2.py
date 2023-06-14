"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, input_dim, 
                 num_classes,
                 transformer_dim=64, 
                 n_heads=8, 
                 dim_feedforward=128, 
                 num_encoder_layers=6, 
                 dropout=0.5, 
                 transformer_activation="gelu", 
                 encode_position = True):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = transformer_dim

        self.input_proj = nn.Sequential(nn.Conv1d(input_dim[1], self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = input_dim[0]
        self.encode_position = encode_position
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = n_heads,
                                       dim_feedforward = dim_feedforward,
                                       dropout = dropout,
                                       activation = transformer_activation)

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = num_encoder_layers,
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes = num_classes
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = data  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # Class probability
        target = self.log_softmax(self.imu_head(target))
        return target

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))



class IMUCLSBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, transformer_dim=64, dropout_prob=0.5):

        super(IMUCLSBaseline, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv1d(input_dim[1], transformer_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(transformer_dim, transformer_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(input_dim[0]*(transformer_dim//2), transformer_dim, nn.ReLU())
        self.fc2 = nn.Linear(transformer_dim,  num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        data = data.swapaxes(1, 2)
        # print(data.shape)
        # # data = data.transpose(1, 2)
        # # print(data.shape)
        x = data
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x) # return B X C/2 x M
        x = x.view(x.size(0), -1) # B X C/2*M
        x = self.fc1(x)
        x = self.log_softmax(self.fc2(x))
        return x # B X N