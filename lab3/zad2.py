import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
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

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

print("Predicting test data")
prediction = clf.predict(test_inputs)
score = metrics.accuracy_score(test_classes, prediction)
print(score)

confusion_matrix = metrics.confusion_matrix(test_classes, prediction, labels=['Setosa', 'Virginica', 'Versicolor'])
print(confusion_matrix)
# Wygrana
# (zaleznosc - przy kazdym uruchomieniu programu drzewo jest generowane na nowo, 
# wiec wyniki bywaja rozne - raz nauczy sie lepiej, raz gorzej)