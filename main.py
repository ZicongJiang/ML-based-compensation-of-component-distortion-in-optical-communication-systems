import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn



import nl_signals


def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zeros_()

# print(model.variables)
def noisy_nlsig(x,N,name,noise_type):
    # gaussian, GMM; noise_type: AWGN
    if name == 'gaussian':
        variance = 1
        mean = 0
        signal = nl_signals.gaussian(x, variance, mean)
    elif name == 'GMM':
        n_componets = 3
        [sep_signal, signal] = nl_signals.gaussian_mixture(x,n_componets)
        plt.figure(0)
        for i in range(n_componets):
            plt.plot(x,sep_signal[i,:],marker='*', markevery=50,label='components %i in gmm' % i,lw=2)
        signal = torch.tensor(signal)
        plt.plot(x,signal,label='gmm_signal')
        plt.xlabel('x')
        plt.ylabel('Nonlinear function')
        plt.xlim([min(x), max(x)])
        plt.grid(visible=True)
        plt.legend()
        plt.savefig('./gmm.png', dpi=600)
        plt.show(block=True)


    if noise_type == 'AWGN':
        mean = torch.zeros(N)
        std = torch.ones(N)
        noise = torch.normal(mean, std/10)

    noisy_signal = signal + noise

    return signal, noisy_signal


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, in_fea, n_hidden, out_fea):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__()
        self.hidden = nn.Linear(in_fea, n_hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n_hidden, out_features=out_fea)
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self,x):
        x = self.relu(self.hidden(x))
        x = self.out(x)
        return x

if __name__=="__main__":
    N = int(1e3)
    epochs = 300
    learning_rate = 0.001
    loss_r = np.zeros(epochs)
    signal_type = 'GMM'

    x = torch.linspace(-10,10,N)
    input_x = torch.rand(N)

    plt.plot(x, input_x, label='input data', lw=.5)
    plt.ylabel('data')
    plt.legend()
    plt.xlim([min(x), max(x)])
    plt.savefig('./input_data.png', dpi=600)
    plt.show(block=True)

    [y, noisy_y] = noisy_nlsig(x, N, signal_type, 'AWGN')  # generate curve we want to fit using NN
    plt.figure(1)
    plt.plot(x, noisy_y, label='nonlinear function with noise',lw=2)
    plt.plot(x, y, label='nonlinear function',lw=2)
    plt.xlabel('x')
    plt.ylabel('Nonlinear function')
    plt.legend()
    plt.xlim([min(x), max(x)])
    plt.grid(visible=True)
    plt.savefig('./signal.png', dpi=600)
    plt.show(block=True)

    net = MLP(N,200,N)
    print(net)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

    plt.ion()

    for step in range(epochs):
        pred = net(input_x)
        loss = loss_func(pred.float(), y.float())
        loss_r[step] = loss.data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epochs: %d, loss: %.8f' % (step, loss.data.numpy()))
        fig = plt.figure(2)
        if step % 20 == 0:
            plt.cla()
            plt.plot(x.data.numpy(), noisy_y.data.numpy(),'b',marker='^',markevery = 100,label = 'noisy signal',lw=2) # target curve
            plt.plot(x.data.numpy(), y.data.numpy(),'y',marker = 'o',markevery = 100,lw=2, label = 'pure signal') # cureve without noisy
            plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=1.5, label = 'learning data') # pred result
            # plt.text(.5, 0, 'Loss=%.8f' % loss.data.numpy())
            plt.legend()
            plt.xlim([min(x),max(x)])
            plt.grid(visible=True)
            plt.pause(0.01)
            plt.savefig('./test%s%i.png' % (signal_type,step), dpi=300)
    plt.ioff()
    plt.figure(3)
    plt.plot(list(range(0,epochs)),loss_r)
    plt.xlim([0, 100])
    plt.xlabel('Epoch Numbers')
    plt.ylabel('Loss')
    plt.grid(visible=True)
    plt.savefig('./iteration.png', dpi=600)
    plt.show(block=True)




