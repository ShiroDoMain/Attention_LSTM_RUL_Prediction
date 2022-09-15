import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import copy


class Transfomer(nn.Module):
    def __init__(self, input_dim=14, input_size=3, hidden_size=2, N=2, output_size=1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.conv = nn.Conv2d(1, 3, kernel_size=(3, 1), stride=1)

        self.lstm = LSTM(input_size=input_dim, hidden_size=hidden_size)
        # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers = 2)

        self.encoder = Encoder(feature_size=3*input_dim, N=N)

        self.lstm_atten = Attention_LSTM(
            input_size=input_size, hidden_size=hidden_size)
        self.channel_atten = Attention_Channel(input_size=input_size)

        self.out = nn.Linear(input_dim*3, out_features=output_size)

    def forward(self, x, t):
        # x -> [1,3,14] -> [1,1,3,14]
        x = x.reshape(1,1,3,self.input_dim)
        o = self.conv(x)

        # o -> [1,3,1,14] -> [1,3,14]
        o = o.reshape(1,3,self.input_dim)

        ca = self.channel_atten(o)
        o = ca * o
        
        o = o.permute(1,0,2)
        o, _ = self.lstm(o)
        ta = self.lstm_atten(o)
        o = ta * o
        
        o = o.permute(1,0,2)
        o = self.encoder(x,t)
        o = self.out(o)
        return o


class Attention_Channel(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.net = nn.Sequential(
            nn.Conv1d(self.input_size, out_channels=10,
                      bias=False, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=self.input_size,
                      bias=False, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.net(self.avg_pool(x))
        max_ = self.net(self.max_pool(avg))
        o = self.sigmoid(avg+max_)
        return o


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # i
        self.w_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.u_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # forget
        self.w_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c
        self.w_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o
        self.w_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def forward(self, x):
        # input_shape = [bs, seq_len, input_dim]
        batch_size, seq_size = x.size(0), x.size(1)

        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        hidden_seq = []
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        for i in range(seq_size):
            input_ = x[:, i, :]
            i_t = torch.sigmoid(input_ @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(input_ @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.tanh(input_ @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(input_ @ self.w_o + h_t @ self.u_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)
        lstm_output = torch.cat(hidden_seq, dim=0)
        lstm_output = lstm_output.view(-1,
                                       hidden_seq[0].shape[0], hidden_seq[0].shape[1])

        return lstm_output, (h_t, c_t)

    def init_weights(self):
        stdv = 1.0 / (self.hidden_size ** self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class Attention_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,  attention_size=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.layer_size = layer_size
        self.attention_size = attention_size

        # self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        # self.lstm = nn.LSTM(input_size, hidden_size)

        self.fc = nn.Sigmoid()

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention(self, x):
        # self attention
        '''
        u_it = tanh(W_w*h_it+b_w)
        α_it = exp(u_it^T * u_w) / Σ_t(exp(u_it^T * u_w))
        s_i = Σ_t(α_it * h_it)
        '''
        # x = x.permute(1,0,2)
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score

        ctx = torch.sum(scored_x, dim=1)
        return ctx

    def forward(self, lstm_out):
        # lstm_out: [bs, seq_len, input_dim]
        out = self.attention(lstm_out)
        out = F.relu(out)
        out = F.relu(out)
        out = self.fc(out)

        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
            (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        pe = np.zeros(self.d_model)

        for i in range(0, self.d_model, 2):
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))

        x = x + Variable(torch.Tensor(pe))
        return x


class Encoder(nn.Module):
    def __init__(self, feature_size=3, N=2, dropout=0.1):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(feature_size)
        self.layers = self.clone(nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=1, dropout=dropout), N)
        self.norm = Norm(feature_size)
        self.feature_size = feature_size

    def clone(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, src, t):
        src = src.reshape(1, self.feature_size)
        x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](x, None)
        return self.norm(x)


if __name__ == "__main__":
    tensor = torch.randn((1, 192, 17))
    # lstm = Transfomer(14)
    lstm = nn.LSTM(17,2,1)
    # lstm = LSTM(17,2)
    out, _ = lstm(tensor)
    # model = Encoder(192*17)
    # out = model(tensor, 0)
    print(out)
    print(out.size())
