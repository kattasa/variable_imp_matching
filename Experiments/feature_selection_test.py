import numpy as np
from sklearn.linear_model import LassoCV, Lasso


x = np.random.normal(0, 1, size=(4000, 100))
beta = np.concatenate([np.ones([10,]), np.zeros([90,])])
u = np.matmul(x, beta)
noise = np.random.normal(0, 0.04, size=(4000,))

y = np.sin(u) + np.sin(2*u) + np.sin(3*u) + 1
# y = np.cos(u) + (1/(1+np.exp(-u))) - u**3 - 2


m = LassoCV().fit(x, y)

print(m.coef_)
print(m.score(x, y))
