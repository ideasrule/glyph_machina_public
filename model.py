from torch import nn
from pyctcdecode import build_ctcdecoder

#For normalization of line images
IMAGE_MEAN = 0.15
IMAGE_STD = 0.38

ALL_CHARS = " -.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¶:"
CHAR_TO_NUM = {c: i + 1 for i, c in enumerate(ALL_CHARS)}
NUM_TO_CHAR = {i + 1: c for i, c in enumerate(ALL_CHARS)}

CTC_DECODER = build_ctcdecoder(
    labels=list(ALL_CHARS),
    kenlm_model_path="bigram_model.arpa"
)

class HTRModel(nn.Module):
    def __init__(self):
        super().__init__()

        #We introduced BatchNorm in anticipation of training with large batch sizes.  However, to minimize
        #VRAM usage and code complexity, we ended up training with a batch size of 1.  The BatchNorm layers
        #therefore likely have no effect.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (4,16), padding=(1,7)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 32, (4,16), padding=(1,7)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 64, (3,8), padding=(1,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(64, 64, (3,8), padding=(1,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
      
        self.lstms = nn.ModuleList([
            nn.LSTM(960, 512, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
            nn.LSTM(1024, 512, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
            nn.LSTM(1024, 512, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
        ])
        self.lin = nn.Linear(1024, len(ALL_CHARS) + 1)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(-1, x.shape[1] * x.shape[2], x.shape[3]).transpose(1,2)
        for layer in self.lstms:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.Dropout1d):
                #Current dimensions are (N,L,C).  We want (N,C,L) so dropout can be per channel. Then we transpose back.
                x = x.transpose(1,2)
                #assert(x.shape[1] == 512) #should be double LSTM hidden size
                x = layer(x)
                x = x.transpose(1,2)
            else:
                assert(False)

        x = self.lin(x)
        x = nn.functional.log_softmax(x, dim=2)
        return x.transpose(1,2)
