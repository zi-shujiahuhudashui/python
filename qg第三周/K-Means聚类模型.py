import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 加载数据
iris = load_iris()
X = iris.data
y_true = iris.target

# 手写 K-Means
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # 随机初始化中心点（从样本中选k个）
        np.random.seed(42)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 分配每个样本到最近的中心
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 计算新中心
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            # 如果中心不再变化，提前结束
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels

# 运行 K-Means
kmeans = KMeans(k=3)
y_pred = kmeans.fit(X)

# 评估（需要真实标签，但聚类是无监督，这里只是验证）
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")