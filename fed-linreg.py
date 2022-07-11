import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import ParameterGrid
from diffprivlib.mechanisms import GaussianAnalytic, Laplace
from diffprivlib.models import LinearRegression 
from copy import deepcopy

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

dtype = torch.float64

# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

d = 120
coeff = torch.randn(size=(d,1), dtype=dtype).to(DEVICE)
noise_multiplier = 0.3

points =  200000
X = 2*(torch.rand(size=(points, d), dtype=dtype).to(DEVICE) - 0.5)
y_true = X@coeff
y = y_true + noise_multiplier*torch.randn_like(y_true)
y_max = y.abs().max()
y = y.div(y_max)

beta = 1e-6
eps = 10

# Sa = torch.tensor(d).sqrt()
Sa_lap = torch.tensor(d)
# Amech = GaussianAnalytic(epsilon=eps/2, delta=(1/points), sensitivity=float(Sa))
Amech = Laplace(epsilon=eps/2, sensitivity=float(Sa_lap))
# Sb = torch.tensor(d**2).sqrt()
Sb_lap = torch.tensor(d**2)
# Bmech = GaussianAnalytic(epsilon=eps/2, delta=(1/points), sensitivity=float(Sb))
Bmech = Laplace(epsilon=eps/2, sensitivity=float(Sb_lap))
DPlinreg = LinearRegression(epsilon=eps, bounds_X=(-1, 1), bounds_y=(-1, 1))

A = X.T@y
B = X.T@X
A_gt = deepcopy(A)
B_gt = deepcopy(B)
reg = beta*torch.eye(B.shape[0], dtype=dtype).to(DEVICE)
B_ = B + reg
A = A.apply_(Amech.randomise).to(DEVICE)
B = B_.apply_(Bmech.randomise).to(DEVICE)
DPlinreg.fit(X.cpu().numpy(), y.cpu().numpy())

Wlsq, res, r, sv = torch.linalg.lstsq(X, y)

Winv= torch.linalg.inv(B_)@A
Wpinv = torch.linalg.pinv(B)@A
Wsolv = torch.linalg.solve(B, A)
Wdp = torch.tensor(DPlinreg.coef_, dtype=dtype, device=DEVICE).view(Winv.shape)

y_pred_inv = X@Winv
y_pred_lsq = X@Wlsq
y_pred_pinv = X@Wpinv
y_pred_solv = X@Wsolv
y_pred_dp = X@Wdp
cost = (y - y_pred_lsq).pow(2).mean().sqrt()

rmse_inv = (y_pred_inv - y).pow(2).mean().sqrt()
rmse_pinv = (y_pred_pinv - y).pow(2).mean().sqrt()
rmse_solv = (y_pred_solv - y).pow(2).mean().sqrt()
rmse_dp = (y_pred_dp - y).pow(2).mean().sqrt()

print(f'rmse wrt least squares: {cost:.3e}')
print(f'pred diff with normal solver: {rmse_inv:.3e}')
print(f'pred diff with pinv solver: {rmse_pinv:.3e}')
print(f'pred diff with linsolve solver: {rmse_solv:.3e}')
print(f'pred diff with dp solver: {rmse_dp:.3e}')

p_lsq = X@Wlsq
p_inv = X@Winv
p_pinv = X@Wpinv
p_solv = X@Wsolv
p_dp = X@Wdp

# plot random N points
N = 20
indices = torch.randint(low=0, high=points, size=(N,))

g = lambda x: torch.index_select(x, 0, indices)

plt.figure()
plt.plot(g(y_true.cpu()), '-.')
plt.plot(y_max*g(y.cpu()), '.')
plt.plot(y_max*g(p_lsq.cpu()), 'x')
plt.plot(y_max*g(p_pinv.cpu()), '--')
plt.plot(y_max*g(p_inv.cpu()), '-x')
plt.plot(y_max*g(p_solv.cpu()), '-x')
# plt.plot(g(p_dp.cpu()), '-x')
plt.legend([f'original', f'gt', f'torch.linalg.lstsq', f'torch.linalg.pinv',
            'torch.linalg.inv', 'torch.linalg.solve']) # ,
            # 'functional dp'])
                
plt.savefig(f'imgs/linreg.png')
plt.close()

# NOTES:
"""
Controlla AnalyseGauss di Dwork (Dp-PCA), FedPCA con DP, e i tre metodi di
Sheffet per "Old Techniques in Differentially Private Linear Regression".
"""