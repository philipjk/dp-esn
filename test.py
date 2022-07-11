import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import ParameterGrid
from diffprivlib.mechanisms import GaussianAnalytic
from diffprivlib.models import LinearRegression 

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

dtype = torch.float64

# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

sequence_points =  2000
sequences = 100
wave_frequency = torch.tensor(2, dtype=dtype)
start = torch.tensor(0.) 
finish = 1*np.pi
exes = torch.arange(start, finish - 1e-6, step=np.pi/sequence_points, dtype=dtype).to(DEVICE)
y = torch.sin(wave_frequency*exes)
ui = exes.view(int(len(exes)*sequences/sequence_points), int(sequence_points/sequences))
uy = y.view(int(len(exes)*sequences/sequence_points), int(sequence_points/sequences)) 
T = len(ui)

def make_W(sparsity: float, # probability of each element to be zeroed-out
           size: tuple, 
           spectral_radius: float=0,
           range: float=1):
    W = 2*range*(1 - torch.rand(size=size, dtype=dtype))
    if sparsity:
        bernoulli_probs = (1-sparsity)*torch.ones_like(W, dtype=dtype)
        sparse_mask = torch.bernoulli(bernoulli_probs)
        W *= sparse_mask 
    if spectral_radius:
        eig = torch.linalg.eigvals(W)
        current_sr = torch.abs(eig).sort(descending=True)[0][0]
        W = W*spectral_radius/current_sr
    return W.to(DEVICE)

def make_readouts(sequences: list,
                  alpha: float,
                  Win: torch.tensor,
                  Ws: torch.tensor):
    readouts = []
    for sequence in sequences:
        x = torch.zeros((Nx, 1), dtype=dtype).to(DEVICE)
        fake_input = torch.tensor((1,), dtype=dtype).to(DEVICE)
        for t, ut in enumerate(sequence):
            leak = (1-alpha)*x
            in_and_bias = torch.cat((ut.view((1,)), fake_input))
            update = alpha*torch.tanh((Win@in_and_bias).view(Nx,1)+ Ws@x)
            x = leak + update
            readouts.append(x)
    X = torch.cat(readouts, 1)
    return X



nx_range =  [50]# [700, 1000, 1200, 1500, 1700, 1900, 2100]
alpha_range = [0.5] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
spars_range = [0.3] # alpha_range
sr_range = [0.9, 0.99] # alpha_range
ws_range = [0.1, 0.5]
beta_range = [1e-7, 1e-5]
costs = []
cost_dict = {}
eps = 100

param_grid = {'nx': nx_range, 'alpha': alpha_range,
              'sparsity': spars_range, 'sr': sr_range,
              'ws_range': ws_range, 'beta_range': beta_range}
grid = ParameterGrid(param_grid)

for i, params in enumerate(grid):
    Nx = params['nx']
    alpha = params['alpha']

    Win = make_W(0., size=(Nx, 1 + 1), range=1)
    Ws = make_W(params['sparsity'], size=(Nx, Nx),
                spectral_radius=params['sr'], range=params['ws_range'])

    X = make_readouts(ui, alpha=alpha, Win=Win, Ws=Ws)

    # inv_arg = X@X.T # + beta*torch.eye(X.shape[0])
    # inv = torch.linalg.inv(inv_arg)

    # Sa = torch.tensor(Nx*sequence_points/sequences).sqrt()
    # Amech = GaussianAnalytic(epsilon=eps, delta=(1/sequences), sensitivity=float(Sa))
    # Sb = torch.tensor((Nx**2)*sequence_points/sequences).sqrt()
    # Bmech = GaussianAnalytic(epsilon=eps, delta=(1/sequences), sensitivity=float(Sb))
    # DPlinreg = LinearRegression(epsilon=eps, bounds_X=(-1, 1), bounds_y=(-1, 1))

    ys = y
    # A = (ys@X.T).cpu()
    # B = (X@X.T).cpu()
    A = (ys@X.T)
    B = (X@X.T)
    reg = params['beta_range']*torch.eye(B.shape[0], dtype=dtype).to(DEVICE)
    B_ = B + reg
    # A = A.apply_(Amech.randomise).to(DEVICE)
    # B = B.apply_(Bmech.randomise).to(DEVICE)
    # DPlinreg.fit(X.T.cpu().numpy(), ys.cpu().numpy())


    W_ = A@torch.linalg.inv(B_)
    Wp = A@torch.linalg.pinv(B)
    W, res, r, sv = torch.linalg.lstsq(X.T, y)
    Wlin = torch.linalg.solve(B, A)
    # Wdp = torch.tensor(DPlinreg.coef_, dtype=dtype, device=DEVICE)

    y_pred_ = X.T@W_

    y_pred = X.T@W
    y_pred_p = X.T@Wp
    y_pred_lin = X.T@Wlin
    # y_pred_dp = X.T@Wdp
    rmse_ = (y_pred_ - y_pred).pow(2).mean().sqrt()
    rmsep = (y_pred_p - y_pred).pow(2).mean().sqrt()
    rmse_lin = (y_pred_lin - y_pred).pow(2).mean().sqrt()
    # rmse_dp = (y_pred_dp - y_pred).pow(2).mean().sqrt()

    print(f'pred diff with normal solver: {rmse_}')
    print(f'pred diff with pinv solver: {rmsep}')
    print(f'pred diff with linsolve solver: {rmse_lin}')
    # print(f'pred diff with dp solver: {rmse_dp}')

    cost = (y_pred - y).pow(2).mean()
    costs.append(cost)
    text = f"""config: nx={params['nx']}, alpha={params['alpha']} 
    sparsity={params['sparsity']}, spectral radius={params['sr']},
    ws_range={params['ws_range']},
    beta={params['beta_range']} cost: {cost:e}"""
    cost_dict[cost] = text
    print(text)
    tryout = int(np.random.randint(0, sequences))
    xx = make_readouts(ui[tryout:tryout+1], params['alpha'], Win, Ws)
    p = xx.T@W
    p_ = xx.T@W_
    pp = xx.T@Wp
    p_lin= xx.T@Wlin
    # p_dp = xx.T@Wdp
    plt.figure()
    plt.plot(uy[tryout].cpu(), 'x')
    plt.plot(p.cpu())
    # plt.plot(p_.cpu())
    plt.plot(pp.cpu())
    plt.plot(p_lin.cpu())
    # plt.plot(p_dp.cpu())
    # plt.legend(['gt', 'torch.linalg.lstsq',
    #              'torch.linalg.inv', 'torch.linalg.pinv', 'torch.linalg.solve',
    #              'dp'])
    plt.legend(['gt', 'torch.linalg.lstsq',
                 'torch.linalg.inv', 'torch.linalg.pinv', 'torch.linalg.solve'])
                 
    plt.title(text)
    plt.savefig(f'imgs/{dtype}-{i}.png')
    plt.close()


print('Results')
minimum = min(costs)
argmin = cost_dict[minimum]
print(argmin)

import pickle
with open('results.pkl', 'wb') as handle:
    pickle.dump(cost_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
