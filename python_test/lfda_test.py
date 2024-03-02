from lfda import lfda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convertToEigenCompatibleMatrix(X : np.ndarray):
    X = X.astype("float64")
    return np.asfortranarray(X)

def convertToEigenCompatibleVector(y : np.ndarray):
    y = y.astype("float64")
    return np.ascontiguousarray(y)


iris_data = load_iris()

X = convertToEigenCompatibleMatrix(iris_data['data'])
y = convertToEigenCompatibleVector(iris_data['target'])

model = lfda.LFDA(n_components=3, k=1, embedding=lfda.EMBEDDING_TYPE.ORTHONORMALIZED)

model.fit(X, y)

x_embed = model.transform(X)

fig = plt.figure(figsize=(12, 6))

# Creating the first subplot for the original data
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0][y == 0], X[:, 1][y == 0], X[:, 2][y == 0], c="red", label="Class 0")
ax1.scatter(X[:, 0][y == 1], X[:, 1][y == 1], X[:, 2][y == 1], c="orange", label="Class 1")
ax1.scatter(X[:, 0][y == 2], X[:, 1][y == 2], X[:, 2][y == 2], c="blue", label="Class 2")
ax1.set_title("Original Data")
ax1.set_xlabel("X axis")
ax1.set_ylabel("Y axis")
ax1.set_zlabel("Z axis")
ax1.legend()

# Creating the second subplot for the transformed/embedded data
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_embed[:, 0][y == 0], x_embed[:, 1][y == 0], x_embed[:, 2][y == 0], c="red", label="Class 0")
ax2.scatter(x_embed[:, 0][y == 1], x_embed[:, 1][y == 1], x_embed[:, 2][y == 1], c="orange", label="Class 1")
ax2.scatter(x_embed[:, 0][y == 2], x_embed[:, 1][y == 2], x_embed[:, 2][y == 2], c="blue", label="Class 2")
ax2.set_title("Transformed/Embedded Data")
ax2.set_xlabel("X axis")
ax2.set_ylabel("Y axis")
ax2.set_zlabel("Z axis")
ax2.legend()

plt.show()

