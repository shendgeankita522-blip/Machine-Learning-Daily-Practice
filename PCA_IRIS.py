from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Example data (replace this with your actual data)
iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the PCA model 
pca = PCA(n_components=2)

# fit and transform the PCA model 
X_pca = pca.fit_transform(X)

# Visualize the reduced-dimensions data 

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1],hue =y, palette='viridis' , s=50)

plt.title('PCA: Iris Dataset')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()