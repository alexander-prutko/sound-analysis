from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t      

# third_tensor = torch.cat((first_tensor, second_tensor), 0)      

class STFT_CQT_Dataset(Dataset):
    def __init__(self, D, C, cqt_transform=None):
        self.stft = D
        self.cqt = C

    def __len__(self):
        d_len = self.stft.shape[1]
        c_len = self.cqt.shape[1]
        assert d_len == c_len, "Lengths of STFT and CQT are not the same"

        return d_len

    def __getitem__(self, index):
        stft = self.stft[:, index]
        cqt = self.cqt[:, index]
        return (stft, cqt)

class STFT2CQT(nn.Module):
    def __init__(self, ngpu):
        super(STFT2CQT, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # state size. 1025
            nn.Linear(1025, 2050),
            nn.BatchNorm1d(2050),
            nn.ReLU(True),
            # state size. 2050
            nn.Linear(2050, 1025),
            nn.BatchNorm1d(1025),
            nn.ReLU(True),
            # state size. 1025
            nn.Linear(1025, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # state size. 512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # state size. 256
            nn.Linear(256, 108),
            nn.Tanh()
            # state size. 108
        )

    def forward(self, input):
        return self.main(input)

def gen_stft2cqt(layers):
    class STFT2CQT_gen(nn.Module):
        def __init__(self, ngpu):
            super(STFT2CQT_gen, self).__init__()
            self.ngpu = ngpu
            self.n_layers = len(layers)
            self.linear = []
            self.bn = []
            for i in range(self.n_layers - 1):
                self.linear.append(nn.Linear(layers[i], layers[i+1]))
            for i in range(self.n_layers - 2):
                self.bn.append(nn.BatchNorm1d(layers[i+1]))

        def forward(self, x):
            for i in range(self.n_layers - 2):
                x = F.relu(self.bbn[i](self.linear[i](x)))
            x = F.tanh(self.linear[-1](x))
            return x

class CQT2STFT(nn.Module):
    def __init__(self, ngpu):
        super(CQT2STFT, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # state size. 108
            nn.Linear(108, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # state size. 256
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # state size. 512
            nn.Linear(512, 1025),
            nn.BatchNorm1d(1025),
            nn.ReLU(True),
            # state size. 1025
            nn.Linear(1025, 2050),
            nn.BatchNorm1d(2050),
            nn.ReLU(True),
            # state size. 2050
            nn.Linear(2050, 1025),
            nn.Tanh()
            # state size. 1025
        )

    def forward(self, input):
        return self.main(input)        

class CQT2STFT_conv(nn.Module):
    def __init__(self, ngpu):
        super(CQT2STFT, self).__init__()
        self.ngpu = ngpu

        self.conv1 = nn.Conv1d(1, 16, 13, padding=6)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(1, 16, 9, dilation=9)
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)

        self.linear5 = nn.Linear(1248, 2050)
        self.bn5 = nn.BatchNorm1d(2050)

        self.linear6 = nn.Linear(2050, 2050)
        self.bn6 = nn.BatchNorm1d(2050)

        self.linear7 = nn.Linear(2050, 1025)

    def forward(self, input): 
        x = F.relu(self.bn1(self.pool1(self.conv1(input)))) # 108 x 1 -> 54 x 16
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))     # 54 x 16 -> 27 x 32

        y = F.relu(self.bn3(self.conv3(input)))             # 108 x 1 -> 12 x 16
        y = F.relu(self.bn4(self.conv4(y)))                 # 12 x 16 -> 12 x 32

        xf = flatten(x) # 864
        yf = flatten(y) # 384

        z = torch.cat((first_tensor, second_tensor))        # 1248

        z = F.relu(self.bn5(self.linear5(z)))               # 2050
        z = F.relu(self.bn6(self.linear6(z)))               # 2050
        z = F.tanh(self.linear7(z))                         # 1025
        return z  