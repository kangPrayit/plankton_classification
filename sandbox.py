import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from skorch.callbacks import EpochScoring
from torch import nn

from skorch import NeuralNetClassifier

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        return X

net = NeuralNetClassifier(
    MyModule,
    max_epochs=200,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    iterator_train__shuffle=True,
    callbacks=[EpochScoring(scoring='accuracy', on_train=True, name='train_acc')],
)
net.fit(X, y)
y_proba = net.predict_proba(X)

plt.plot(net.history[:, 'train_acc'], label='train accuracy')
plt.plot(net.history[:, 'valid_acc'], label='valid accuracy')
plt.legend()
plt.show()