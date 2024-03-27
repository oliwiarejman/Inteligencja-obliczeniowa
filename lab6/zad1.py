import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

df['Age'] = pd.cut(df['Age'], bins=[0, 18, 65, 99], labels=['Child', 'Adult', 'Elderly'])

df['Survived'] = df['Survived'].map({'No': 0, 'Yes': 1})


