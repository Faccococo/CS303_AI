from project3.NNet import NNet
from project3.Model_train.DataSet import dataSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os


lr = 0.005
epochs = 1000
batch_size = 20

cuda = True if torch.cuda.is_available() else False


def train(model, features_train, labels_train):
    model.classifier.train()
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr)
    dataset = dataSet(features_train, labels_train)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    counter = 0

    for epoch in range(epochs):
        counter = 0
        for data in dataloader:
            if cuda:
                data.cuda()
            feature, label = data

            prediction = model.classifier(feature).squeeze(-1)
            prediction = prediction.float()
            label = label.float()

            loss = loss_function(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, epochs, counter, len(features_train), loss.item())
            )
            counter += len(data[0])


def get_score(model, features_test, labels_test):
    model.classifier.eval()
    right = 0
    fail = 0
    for i in range(len(features_test)):
        feature = features_test[i]
        label = labels_test[i]
        outputs = torch.round(model.classifier(feature))[0]
        # print('Output: ', outputs, '  ', 'label: ', label)
        if int(outputs) == int(label):
            right += 1
        else:
            fail += 1
    print("model score: ", right / (right + fail))


data = torch.load("project3/Model_train/data.pth")
label = data["label"]
feature = data["feature"]

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2
)


if not os.path.exists('project3/model_d.pth'):
    model = NNet()
    train(model, feature_train, label_train)
    torch.save(model.state_dict(), "project3/model_d.pth")
model = NNet()
model.load_state_dict(torch.load("project3/model_d.pth"))
get_score(model, feature_test, label_test)
