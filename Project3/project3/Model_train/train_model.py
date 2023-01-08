from project3.agent import NNet
from project3.Model_train.DataSet import dataSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os


lr = 0.02
epochs = 100
batch_size = 10
lr_ratio = 0.5

cuda = True if torch.cuda.is_available() else False


def train(model, features_train, labels_train):
    model.classifier.train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), lr=lr * lr_ratio)
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
            prediction = torch.FloatTensor(
                model.classifier(feature).squeeze(-1))

            loss = loss_function(prediction, label)

            if loss.item() > 0.5:
                optimizer = torch.optim.SGD(
                    model.classifier.parameters(), lr=0.02 * lr_ratio)
            elif loss.item() > 0.1:
                optimizer = torch.optim.SGD(
                    model.classifier.parameters(), lr=0.01 * lr_ratio)
            elif loss.item() > 0.05:
                optimizer = torch.optim.SGD(
                    model.classifier.parameters(), lr=0.005 * lr_ratio)
            elif loss.item() > 0.01:
                optimizer = torch.optim.SGD(
                    model.classifier.parameters(), lr=0.002 * lr_ratio)
            else:
                optimizer = torch.optim.SGD(
                    model.classifier.parameters(), lr=0.0005 * lr_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
            #     % (epoch, epochs, counter / batch_size, len(features_train) / batch_size, loss.item())
            # )
            print(
                "[Epoch %d/%d] [Data Sizes %d] [Loss: %f]"
                % (epoch, epochs, len(features_train), loss.item())
            )
            counter += len(data[0])


def get_score(model, features_test, labels_test):
    model.classifier.eval()
    right = 0
    fail = 0
    for i in range(len(features_test)):
        feature = features_test[i]
        label = labels_test[i]
        outputs = model.get_result(feature)
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
    torch.save(feature_test, "project3/test_feature.pth")
    torch.save(label_test, "project3/test_label.pth")
else:
    feature_test = torch.load("project3/test_feature.pth")
    label_test = torch.load("project3/test_label.pth")
model = NNet()
model.load_state_dict(torch.load("project3/model_d.pth"))
get_score(model, feature_test, label_test)

# model = NNet()
# train(model, feature_train, label_train)
# get_score(model, feature_test, label_test)
