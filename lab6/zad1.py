import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

df['Survived'] = df['Survived'].map({'No': 0, 'Yes': 1})

df['Survived'] = df['Survived'].apply(lambda x: 'No' if x == 0 else 'Yes')

df = df.drop(df.columns[0], axis=1)

basket_sets = pd.get_dummies(df)

frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

rules = rules.sort_values(['confidence'], ascending=False)

top_10_rules = rules.head(10)

plt.figure(figsize=(10, 7))
plt.barh(range(len(top_10_rules)), top_10_rules['confidence'], color='skyblue', alpha=0.6)
plt.yticks(range(len(top_10_rules)), top_10_rules['antecedents'].apply(lambda x: ', '.join(x)).values)
plt.xlabel('Ufność')
plt.ylabel('Reguły')
plt.title('10 najciekawszych reguł dotyczących przeżywalności na Titanicu')
plt.gca().invert_yaxis()
plt.show()

print("10 najciekawszych reguł dotyczących przeżywalności:")
print(top_10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
