import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    os.mkdir('pictures')
except Exception as e:
    pass
filenames = os.listdir('results')
for name in filenames:
    print('ploting:', name)
    df = pd.read_csv(os.path.join('results', name), names=['predict', 'fact'])
    prefix = '.'.join(name.split('.')[:-1])
    error = (df['predict'] - df['fact']).abs()
    rel_error_pd = error / df['fact']
    rel_error = np.sort(rel_error_pd)
    size = len(rel_error)
    plt.plot(rel_error[:int(size * 0.99)])
    plt.text(0, 0.5, rel_error_pd.describe())
    plt.title('CDF of relative error of 99% data')
    plt.ylabel('relative error')
    plt.xlabel('number of sample')
    plt.savefig(os.path.join('pictures', prefix + '.png'), format='png')
    plt.cla()
