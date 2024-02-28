import pandas as pd
import difflib

df = pd.read_csv("iris_with_errors.csv")

print(df.info())
print(df.describe(include='all'))


for column in df.columns[:-1]:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    col_mean = df[column].mean() 
    df[column] = df[column].fillna(col_mean) 
    

print(df)

valid_varieties = ["Setosa", "Versicolor", "Virginica"]

def find_similar(variety):
    matches = difflib.get_close_matches(variety.capitalize(), valid_varieties)
    if matches:
        return matches[0]
    else:
        return variety

df['variety'] = df['variety'].apply(find_similar)

print(df['variety'].unique())

df.to_csv("iris_fixed.csv", index=False)
