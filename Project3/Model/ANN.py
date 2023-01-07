from NNet import NNet
import torch
from torch.autograd import Variable
from DataSet import dataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import os

cuda = True if torch.cuda.is_available() else False

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.ReLU()
        )

    def forward(self, feature):
        result = self.model(feature)
        return result



class ANN:
    lr = 0.005
    epochs = 1000
    batch_size = 20

    def __init__(self):
        self.model = NNet()
        if cuda: self.model.cuda()


    def train(self, features_train, labels_train):
        self.model.train()
        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        dataset = dataSet(features_train, labels_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
        counter = 0

        for epoch in range(self.epochs):
            counter = 0
            for data in dataloader:
                if cuda: data.cuda()
                feature, label = data

                prediction = self.model(feature).squeeze(-1)
                prediction = prediction.float()
                label = label.float()

                loss = loss_function(prediction, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                    % (epoch, self.epochs, counter, len(features_train), loss.item())
                )
                counter += len(data[0])



    def get_score(self, features_test, labels_test):
        self.model.eval()
        right = 0
        fail = 0
        for i in range(len(features_test)):
            feature = features_test[i]
            label = labels_test[i]
            outputs = torch.round(self.model(feature))[0]
            # print('Output: ', outputs, '  ', 'label: ', label)
            if int(outputs) == int(label):
                right += 1
            else: fail += 1
        print("model score: ", right / (right + fail))
            
from sklearn.model_selection import train_test_split
import torch

data = torch.load("data.pth")
label = data["label"]
feature = data["feature"]

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2
)



if not os.path.exists('model.pt'):
    model = ANN()
    model.train(feature_train, label_train)
    torch.save(model, "model.pt")
model = torch.load("model.pt")
model.get_score(feature_test, label_test)

# model = ANN()
# model.train(feature_train, label_train)
