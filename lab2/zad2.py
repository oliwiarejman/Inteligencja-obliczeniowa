from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def plot_pca_iris(data):
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="FlowerType")
    
    pca_data = PCA(n_components=2).fit(data.data)
    print(pca_data)
    print(pca_data.explained_variance_ratio_)
    print(pca_data.components_)
    print(pca_data.transform(data.data))

    tran_data = pca_data.transform(data.data)
    
    tran_data = pd.DataFrame(tran_data, columns=['PC1', 'PC2'])
    tran_data = pd.concat([tran_data, y], axis=1)
    
    plt.figure(figsize=(8, 6))
    for flower_type in y.unique():
        plt.scatter(tran_data.loc[tran_data['FlowerType'] == flower_type, 'PC1'],
                    tran_data.loc[tran_data['FlowerType'] == flower_type, 'PC2'],
                    label=f'Class {flower_type}')
    
    plt.title('PCA of Iris Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

iris = datasets.load_iris()
plot_pca_iris(iris)
