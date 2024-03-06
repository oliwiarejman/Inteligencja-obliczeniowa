import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

data = datasets.load_iris()
data_df = pd.DataFrame(data.data, columns=data.feature_names)

sepal_data = data_df[['sepal length (cm)', 'sepal width (cm)']].copy()

original_sepal_length_min = sepal_data['sepal length (cm)'].min()
original_sepal_length_max = sepal_data['sepal length (cm)'].max()
original_sepal_length_mean = sepal_data['sepal length (cm)'].mean()
original_sepal_length_std = sepal_data['sepal length (cm)'].std()

original_sepal_width_min = sepal_data['sepal width (cm)'].min()
original_sepal_width_max = sepal_data['sepal width (cm)'].max()
original_sepal_width_mean = sepal_data['sepal width (cm)'].mean()
original_sepal_width_std = sepal_data['sepal width (cm)'].std()

scaler_minmax = MinMaxScaler()
sepal_data_minmax = scaler_minmax.fit_transform(sepal_data)

sepal_data_minmax_df = pd.DataFrame(sepal_data_minmax, columns=['sepal length (cm)', 'sepal width (cm)'])

scaler_zscore = StandardScaler()
sepal_data_zscore = scaler_zscore.fit_transform(sepal_data)

sepal_data_zscore_df = pd.DataFrame(sepal_data_zscore, columns=['sepal length (cm)', 'sepal width (cm)'])

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(sepal_data['sepal length (cm)'], sepal_data['sepal width (cm)'], c=data.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Oryginalne Sepal Data')

plt.subplot(1, 3, 2)
plt.scatter(sepal_data_minmax_df['sepal length (cm)'], sepal_data_minmax_df['sepal width (cm)'], c=data.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Min-Max Scaling Sepal Data')

plt.subplot(1, 3, 3)
plt.scatter(sepal_data_zscore_df['sepal length (cm)'], sepal_data_zscore_df['sepal width (cm)'], c=data.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Z-Score Scaling Sepal Data')

plt.tight_layout()
plt.show()

print("Wyniki dla Oryginalnych Danych:")
print("-"*30)
print("Długość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(original_sepal_length_min))
print("  - Maksymalna wartość: {:.2f}".format(original_sepal_length_max))
print("  - Średnia wartość: {:.2f}".format(original_sepal_length_mean))
print("  - Odchylenie standardowe: {:.2f}".format(original_sepal_length_std))
print("")
print("Szerokość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(original_sepal_width_min))
print("  - Maksymalna wartość: {:.2f}".format(original_sepal_width_max))
print("  - Średnia wartość: {:.2f}".format(original_sepal_width_mean))
print("  - Odchylenie standardowe: {:.2f}".format(original_sepal_width_std))
print("")

print("Wyniki dla Danych po Min-Max Scaling:")
print("-"*30)
print("Długość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(sepal_data_minmax_df['sepal length (cm)'].min()))
print("  - Maksymalna wartość: {:.2f}".format(sepal_data_minmax_df['sepal length (cm)'].max()))
print("  - Średnia wartość: {:.2f}".format(sepal_data_minmax_df['sepal length (cm)'].mean()))
print("  - Odchylenie standardowe: {:.2f}".format(sepal_data_minmax_df['sepal length (cm)'].std()))
print("")
print("Szerokość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(sepal_data_minmax_df['sepal width (cm)'].min()))
print("  - Maksymalna wartość: {:.2f}".format(sepal_data_minmax_df['sepal width (cm)'].max()))
print("  - Średnia wartość: {:.2f}".format(sepal_data_minmax_df['sepal width (cm)'].mean()))
print("  - Odchylenie standardowe: {:.2f}".format(sepal_data_minmax_df['sepal width (cm)'].std()))
print("")

print("Wyniki dla Danych po Z-Score Scaling:")
print("-"*30)
print("Długość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(sepal_data_zscore_df['sepal length (cm)'].min()))
print("  - Maksymalna wartość: {:.2f}".format(sepal_data_zscore_df['sepal length (cm)'].max()))
print("  - Średnia wartość: {:.2f}".format(sepal_data_zscore_df['sepal length (cm)'].mean()))
print("  - Odchylenie standardowe: {:.2f}".format(sepal_data_zscore_df['sepal length (cm)'].std()))
print("")
print("Szerokość działki kielicha:")
print("  - Minimalna wartość: {:.2f}".format(sepal_data_zscore_df['sepal width (cm)'].min()))
print("  - Maksymalna wartość: {:.2f}".format(sepal_data_zscore_df['sepal width (cm)'].max()))
print("  - Średnia wartość: {:.2f}".format(sepal_data_zscore_df['sepal width (cm)'].mean()))
print("  - Odchylenie standardowe: {:.2f}".format(sepal_data_zscore_df['sepal width (cm)'].std()))
