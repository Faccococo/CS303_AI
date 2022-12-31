from sklearn.model_selection import train_test_split
import torch
from sklearn.linear_model import LogisticRegression

data = torch.load("data.pth")
label = data["label"]
feature = data["feature"]

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2
)

model = LogisticRegression()
model.fit(feature_train, label_train)
model.score(feature_test, label_test)
