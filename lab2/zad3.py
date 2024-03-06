import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

data = datasets.load_iris()
data_df = pd.DataFrame(data.data, columns=data.feature_names)

sepal_data = data_df[['sepal length (cm)', 'sepal width (cm)']].copy()

scaler_minmax = MinMaxScaler()
sepal_data_minmax = scaler_minmax.fit_transform(sepal_data)

scaler_zscore = StandardScaler()
sepal_data_zscore = scaler_zscore.fit_transform(sepal_data)

sepal_data_minmax_df = pd.DataFrame(sepal_data_minmax, columns=['sepal length (cm)', 'sepal width (cm)'])
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