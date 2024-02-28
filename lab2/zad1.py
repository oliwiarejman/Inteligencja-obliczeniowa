import pandas as pd

df = pd.read_csv("iris_with_errors.csv")

print(df.info())
print(df.describe(include='all'))


for column in df.columns[:-1]:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    col_mean = df[column].mean() 
    df[column] = df[column].fillna(col_mean) 
    

print(df)

df['variety'] = df['variety'].str.capitalize() 


print(df['variety'].unique())

df['variety'] = df['variety'].replace({"Versicolour": "Versicolor", "Virginicaa": "Virginica"})

print(df['variety'].unique())

df.to_csv("iris_fixed.csv", index=False)
