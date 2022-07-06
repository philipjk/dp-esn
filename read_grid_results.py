import matplotlib.pyplot as plt
import pandas as pd

# read and filter data
data = pd.read_csv('grid-search-screenlog.0', sep='(\d+(?:\.\d+)?)', engine='python', header=None)
data = data.iloc[:, [1,3,5,7,9]]
columns = {1: 'nx', 3: 'alpha', 5: 'sparsity', 7: 'radius', 9: 'cost'}
data.rename(columns=columns, inplace=True)
data = data[data['cost']<1e4]
print(data.head())

fig, ax = plt.subplots(nrows=2, ncols=2)
axes = ax.ravel()
for i, col in enumerate(data):
    print(i, col)
    if col=='cost':
        continue
    line = data.groupby(by=col).mean()['cost']
    axes[i].plot(line)
    axes[i].set_ylabel(f'cost')
    axes[i].set_xlabel(f'{col}')

plt.tight_layout()
plt.savefig('gs-results.png')

