import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
encoder = LabelEncoder()
scaler = StandardScaler()
df = pd.read_csv("diabetes.csv")
# podzial na zbior testowy 0% i treningowy 70% z ziarnem 13
# podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=281195)

train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]

scaler.fit(train_inputs)
scaler.fit(test_inputs)
train_inputs = scaler.transform(train_inputs)
test_inputs = scaler.transform(test_inputs)
test_classes = test_classes # encoder.fit_transform(test_classes)
train_classes = train_classes # encoder.fit_transform(train_classes)
# adam ma wiecej na testowym o 0.005 ale mocno pada na treningowym, 
# lbfgs jest bardziej wyposrodkowany, co do activation to relu wypadlo najlepiej
clf = MLPClassifier(solver="lbfgs",
                    random_state=0,
                    activation="relu",
                    hidden_layer_sizes=(6, 3,),
                    max_iter=5000)

clf.fit(train_inputs, train_classes)

score_train = clf.score(train_inputs, train_classes)
score_test = clf.score(test_inputs, test_classes)

predictions_train = clf.predict(train_inputs)
predictions_test = clf.predict(test_inputs)
# print(predictions_test)
accuracy_train = accuracy_score(predictions_train, train_classes)
accuracy_test = accuracy_score(predictions_test, test_classes)
print("Accuracy score train & test: ", accuracy_train, " & ", accuracy_test)
confusion_matrix = confusion_matrix(test_classes, predictions_test, labels=clf.classes_ )
train_inputs = scaler.transform(train_inputs)
test_inputs = scaler.transform(test_inputs)
print(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
disp.plot()
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig("confusion_plot.png")
