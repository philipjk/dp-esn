import numpy as np

np.random.seed(123)
sequences = 600000
sequence_len = 10
res_size = 300
X = np.random.uniform(-1, 1, size=(res_size, sequences*sequence_len))

W = 4*np.ones(shape=(1, res_size))
Y = W@X +  0.1*np.random.randn(1, sequences*sequence_len)
Y_scaler = np.abs(Y).max()
Y_scaled = Y/Y_scaler
print(Y_scaled.min(), Y_scaled.max())

eps = 1

A = Y_scaled@X.T + np.random.normal(loc=0, scale=sequence_len*res_size, size=(1, res_size))
B = X@X.T + np.random.laplace(loc=0, scale=sequence_len*res_size*res_size, size=(1, res_size))
West = A@np.linalg.inv(B) # + 0.01*np.identity(res_size))

print(np.sum(np.power(Y-Y_scaler*(West@X), 2))/Y.size)
