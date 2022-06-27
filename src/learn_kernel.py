import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


class BatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LearningKernel(nn.Module):
    def __init__(self, input_dim, output_dim, num_rand_features, interpret=True):
        super(LearningKernel, self).__init__()
        torch.manual_seed(888)
        if interpret:
            self.log_kernel_weight = nn.Parameter(torch.randn(input_dim), requires_grad=True)
            self.omega = torch.randn((input_dim, num_rand_features))
        else:
            self.log_kernel_weight = None
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.omega = torch.randn((16, num_rand_features))
        self.D = num_rand_features
        self.b = torch.rand(num_rand_features) * math.pi * 2
        self.rff = nn.Linear(num_rand_features, output_dim)

    def trans_to_rff(self, x):
        """
        Transform X features to Z space given optimized model, omega and b
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(x, dtype=torch.float, device=device)
        with torch.no_grad():
            if self.log_kernel_weight is not None:
                x = x * torch.exp(self.log_kernel_weight)
            else:
                x = torch.tanh(self.fc1(x))
                x = self.fc2(x)
            z = math.sqrt(2 / self.D) * torch.cos((torch.mm(x, self.omega) + self.b))
        return z.numpy()

    def forward(self, x):
        if self.log_kernel_weight is not None:
            x = x * torch.exp(self.log_kernel_weight)
        else:
            x = torch.tanh(self.fc1(x))
            x = self.fc2(x)
        x = math.sqrt(2 / self.D) * torch.cos((torch.mm(x, self.omega) + self.b))
        x = self.rff(x)
        return F.softmax(x, dim=1)

    def get_feature_importance(self):
        with torch.no_grad():
            return torch.exp(self.log_kernel_weight).numpy()


def train_kernel(X_train, y_train, num_classes, loss_weight=None, interpret=True,
                 num_epochs=300, num_rand_features=500, lr=0.01,
                 batch_size=64, verbose=True):
    """
    Return the best kernel weights in the original feature space
    @:param
        X_train: Features in original forms
        y_train: Labels
    @:return
        Learned model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    model = LearningKernel(X_train.size(1), num_classes, num_rand_features, interpret=interpret).to(device)
    data_loader = DataLoader(BatchDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Set up cross entropy loss
    if loss_weight is not None:
        loss_weight = torch.tensor(loss_weight, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train kernel
    model.train()
    for epoch in range(num_epochs):
        acc_loss = []
        for data, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            acc_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        if verbose and epoch % 20 == 0:
            print('Epoch {0}/{1}, Learning Kernel Loss: {2}'.format(epoch, num_epochs, sum(acc_loss) / len(acc_loss)))

    return model


def eval_auc_acc(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float, device=device)
    model.eval()
    y_pred = model(X).detach().argmax(dim=1).numpy()
    return multiclass_roc_auc_score(y, y_pred), accuracy_score(y, y_pred)


def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
