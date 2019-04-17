import torch.utils.data.Dataset

class STFT_CQT_Dataset(Dataset):
    def __init__(self, D, C):
        self.stft = D
        self.cqt = C

    def __len__(self):
        d_len = D.shape[1]
        c_len = C.shape[1]
        assert d_len == c_len, "Lengths of STFT and CQT are not the same"

        return d_len

    def __getitem__(self, index):
        return (self.stft[:,index], self.cqt[:,index])