import torch.utils.data.Dataset
import torch.nn as nn

class STFT_CQT_Dataset(Dataset):
    def __init__(self, D, C, stft_transform, cqt_transform):
        self.stft = D
        self.cqt = C
        self.stft_transform = stft_transform
        self.cqt_transform = cqt_transform

    def __len__(self):
        d_len = D.shape[1]
        c_len = C.shape[1]
        assert d_len == c_len, "Lengths of STFT and CQT are not the same"

        return d_len

    def __getitem__(self, index):
        stft = self.stft[:, index]
        cqt = self.cqt[:, index]
        if self.stft_transform is not None:
            stft = self.stft_transform(stft)
        if self.cqt_transform is not None:
            cqt = self.cqt_transform(cqt)
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