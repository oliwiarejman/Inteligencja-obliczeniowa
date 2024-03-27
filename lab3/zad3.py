import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import graphviz
df = pd.read_csv("iris.csv")
# podzial na zbior testowy 0% i treningowy 70% z ziarnem 13
# podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=281195)

print(test_set)
print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)
print(clf.predict([[5.1, 3.8, 1.9, 0.4]]))
tree.plot_tree(clf)

predictionTree = clf.predict(test_inputs)
scoreTree = metrics.accuracy_score(test_classes, predictionTree)
confusion_matrix = metrics.confusion_matrix(test_classes, predictionTree, labels=['Setosa', 'Virginica', 'Versicolor'])
print("Score for tree", scoreTree, "\n", confusion_matrix)

n3 = KNeighborsClassifier(n_neighbors=3)
n3.fit(train_inputs, train_classes)
prediction3 = n3.predict(test_inputs)
score3 = metrics.accuracy_score(test_classes, prediction3)
confusion_matrix = metrics.confusion_matrix(test_classes, prediction3, labels=['Setosa', 'Virginica', 'Versicolor'])
print("Score for KNN3", score3, "\n", confusion_matrix)

n5 =  KNeighborsClassifier(n_neighbors=5)
n5.fit(train_inputs, train_classes)
prediction5 = n5.predict(test_inputs)
score5 = metrics.accuracy_score(test_classes, prediction5)
confusion_matrix = metrics.confusion_matrix(test_classes, prediction5, labels=['Setosa', 'Virginica', 'Versicolor'])
print("Score for KNN5", score5, "\n", confusion_matrix)

n11 = KNeighborsClassifier(n_neighbors=11)
n11.fit(train_inputs, train_classes)
prediction11 = n11.predict(test_inputs)
score11 = metrics.accuracy_score(test_classes, prediction11)
confusion_matrix = metrics.confusion_matrix(test_classes, prediction11, labels=['Setosa', 'Virginica', 'Versicolor'])
print("Score for KNN11", score11, "\n", confusion_matrix)

naive_bayes = GaussianNB()
naive_bayes.fit(train_inputs, train_classes)
predictionNB = naive_bayes.predict(test_inputs)
scoreNB = metrics.accuracy_score(test_classes, predictionNB)
confusion_matrix = metrics.confusion_matrix(test_classes, predictionNB, labels=['Setosa', 'Virginica', 'Versicolor'])
print("Score for naive bayes", scoreNB, "\n", confusion_matrix)















