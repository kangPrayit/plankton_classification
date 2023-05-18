from skorch import NeuralNet
from skorch.callbacks import EpochScoring, Freezer
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import *
from model import *
from metrics import *

if __name__ == '__main__':
    train_dir = './datasets/plankton_cocov2/train/'
    test_dir = './datasets/plankton_cocov2/test/'
    # train_ds = PlanktonDataset(train_dir, transform=transforms.ToTensor())
    # test_ds = PlanktonDataset(test_dir, transform=transforms.ToTensor())
    train_ds = PlanktonSegmentationDataset(data_dir=train_dir, transform=transforms.ToTensor())
    # batch_size = 16
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_ds, batch_size=batch_size)

    # model = UNetResNetAttention(in_channels=3, out_channels=1, backbone='resnet34', pretrained=True)
    # model = UNet(in_channels=3, out_channels=1)
    model = UNet(pretrained=True)
    freezer = Freezer('conv*')
    criterion = nn.BCEWithLogitsLoss
    optimizer = optim.Adam
    net = NeuralNet(
        model,
        max_epochs=10,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=16,
        iterator_train__shuffle=True,
        # iterator_valid=test_ds,
        callbacks=[
            freezer,
            EpochScoring(scoring=dice_coefficient, name='dice', lower_is_better=False),
            EpochScoring(scoring=iou, name='iou', lower_is_better=False)
        ],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    net.fit(train_ds, y=None)
