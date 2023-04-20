import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
from net_optimizor import net_optimizor

import nl_signals

# print(model.variables)
def noisy_nlsig(x,N,name,noise_type,std):
    # gaussian, GMM; noise_type: AWGN
    if name == 'gaussian':
        variance = 1
        mean = 0
        signal = nl_signals.gaussian(x, variance, mean)
    elif name == 'GMM':
        n_componets = 3
        [sep_signal, signal] = nl_signals.gaussian_mixture(x,n_componets)
        # plt.figure(0)
        # for i in range(n_componets):
        #     plt.plot(x,sep_signal[i,:],marker='*', markevery=50,label='components %i in gmm' % i,lw=2)
        signal = torch.tensor(signal)
        # plt.plot(x,signal,label='gmm_signal')
        # plt.xlabel('x')
        # plt.ylabel('Nonlinear function')
        # plt.xlim([min(x), max(x)])
        # plt.grid(visible=True)
        # plt.legend()
        # plt.savefig('./gmm.png', dpi=600)
        # plt.show(block=True)


    if noise_type == 'AWGN':
        mean = torch.zeros(N)
        std_int = torch.ones(N)
        std = std_int * std
        noise = torch.normal(mean, std)

    noisy_signal = signal + noise

    return signal, noisy_signal

if __name__=="__main__":

    lr_n = 50
    epochs = 1000
    learning_rate_range = np.linspace(0.0001,0.5,lr_n)
    activation_function = ['Sigmoid', 'ReLU', 'Tanh']
    N = int(2e3)
    std = 0.2
    nodes_num_range = range(1, 20+1)

    [learning_rate_train, nodes_num_train, learning_val, nodes_num_val] = net_optimizor(std,N,epochs,learning_rate_range,nodes_num_range)




