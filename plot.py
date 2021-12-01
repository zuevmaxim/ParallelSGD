import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('w8a.txt', names=['name', 'threads', 'time_ms', 'prec', 'mse'])
for target in ['time_ms', 'prec', 'mse']:
    for name in np.unique(df['name']):
        df_ = df[df['name'] == name]
        sns.lineplot(x=df_['threads'], y=df_[target], label=name)
    plt.title(target)
    plt.legend()
    plt.show()
