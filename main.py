from collections import OrderedDict

from sklearn.model_selection import cross_val_score, GridSearchCV
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split, SliceDataset
from torch import nn, optim

from torchvision import transforms, datasets



# DL Model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 6, 5)),
                ('relu', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(2, 2)),
                ('conv2', nn.Conv2d(6, 16, 5)),
                ('maxpool2', nn.MaxPool2d(2, 2)),
                ('relu', nn.ReLU()),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(44944, 120)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(120, 84)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(84, 13))
            ])
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Configuration
    lr = 0.01
    epochs = 5
    data_dir = "./datasets/09042023_HABS"
    # data_dir = "/content/drive/MyDrive/DEEP LEARNING PROJECT/DATASETS/09042023_HABS"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    train_ds = datasets.ImageFolder(data_dir, train_transform)

    net = NeuralNetClassifier(
        MyModel,
        criterion=nn.CrossEntropyLoss,
        lr=lr,
        batch_size=4,
        max_epochs=epochs,
        train_split=None,
        optimizer=optim.Adam,
        device='cuda',
    )
    # net.fit(train_ds, y=None)

    train_sliceable = SliceDataset(train_ds)
    # scores = cross_val_score(net, train_sliceable, y=None, cv=5, scoring='accuracy')
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=5, scoring='accuracy')
    gs.fit(train_ds, y=None)
    print(gs.best_score_, gs.best_params_)
