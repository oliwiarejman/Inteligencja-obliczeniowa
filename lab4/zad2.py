import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=281195)
# print(df.values)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
# print(test_classes)
scaler.fit(train_inputs)
train_inputs = scaler.transform(train_inputs)
test_inputs = scaler.transform(test_inputs)
# labels = scaler.transform(test_classes)
# print("LABELS", labels)
# print(train_inputs)
# hidden_layer_sizes=(2,) 
# hidden_layer_sizes=(3,) 
# hidden_layer_sizes=(3,3,) prawie 100%

clf = MLPClassifier(solver="lbfgs",
                    alpha=1e-5,
                    random_state=1,
                    hidden_layer_sizes=(3,),
                    max_iter=10000)

clf.fit(train_inputs, train_classes)

score_train = clf.score(train_inputs, train_classes)
score_test = clf.score(test_inputs, test_classes)

predictions_train = clf.predict(train_inputs)
predictions_test = clf.predict(test_inputs)
#print(predictions_test)
accuracy_train = accuracy_score(predictions_train, train_classes)
accuracy_test = accuracy_score(predictions_test, test_classes)
print("Accuracy score train & test: ", accuracy_train, " & ", accuracy_test)
# print(clf.n_features_in_)
