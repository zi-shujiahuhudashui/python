import numpy as np
import pandas as pd

# 1. 加载数据
df = pd.read_csv('wine_data.csv', sep=';')
X = df.drop('quality', axis=1).values   # 所有特征
y_reg = df['quality'].values            # 回归目标（评分）
y_bin = (y_reg > 6).astype(int)         # 分类目标（好酒=1）

# 2. 手写数据集划分（70% 训练，30% 测试）
np.random.seed(42)   # 固定随机种子，让结果可复现
indices = np.random.permutation(len(X))   # 随机打乱索引
train_size = int(0.7 * len(X))
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_raw = X[train_idx]
X_test_raw = X[test_idx]
y_reg_train = y_reg[train_idx]
y_reg_test = y_reg[test_idx]
y_bin_train = y_bin[train_idx]
y_bin_test = y_bin[test_idx]

# 3. 手写标准化（减去均值，除以标准差）
def standardize(X_train, X_test):
    """用训练集的均值和标准差来标准化训练集和测试集"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # 防止除以 0
    std[std == 0] = 1
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

X_train, X_test = standardize(X_train_raw, X_test_raw)

# 4. 线性回归（梯度下降）
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        """R² 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

# 训练线性回归
lr_model = LinearRegressionGD(lr=0.01, n_iters=2000)
lr_model.fit(X_train, y_reg_train)
r2 = lr_model.score(X_test, y_reg_test)
print(f"线性回归 R²: {r2:.4f}")

# 5. 逻辑回归（梯度下降）
class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# 训练逻辑回归
log_model = LogisticRegressionGD(lr=0.1, n_iters=3000)
log_model.fit(X_train, y_bin_train)
acc = log_model.accuracy(X_test, y_bin_test)
print(f"逻辑回归准确率: {acc:.4f}")