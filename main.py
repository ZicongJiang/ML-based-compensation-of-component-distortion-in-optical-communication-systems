import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
import nl_signals

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

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


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, in_fea, n_hidden, out_fea):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_fea, n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.active = nn.Sigmoid()
        self.out = nn.Linear(n_hidden, out_features=out_fea)
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self,x):
        x = self.active(self.hidden1(x))
        x = self.active(self.hidden2(x))
        x = self.active(self.hidden3(x))
        x = self.out(x)
        return x

if __name__=="__main__":
    # lr_n = 5
    epochs = 1000
    learning_rate = 0.04090816
    # learning_rate_seq = np.linspace(0.001,0.1,lr_n)
    # loss_val_r_l = np.zeros([lr_n, epochs])
    # loss_r_l = np.zeros([lr_n, epochs])

    # for lr in range(lr_n):
    N = int(2e3)
    std = 0.2
    nodes_num = 17
    loss_r = np.zeros(epochs)

    loss_val_r = np.zeros(epochs)
    signal_type = 'GMM'
    cross_validation = 0.2
    train_l = N * (1 - cross_validation)
    val_l = N * cross_validation

    x = torch.linspace(-10,10,N)
    input_x = torch.rand(N)

    # plt.plot(x, input_x, label='input data', lw=.5)
    # plt.ylabel('data')
    # plt.legend()
    # plt.xlim([min(x), max(x)])
    # plt.savefig('./input_data.png', dpi=600)
    # plt.show(block=True)

    [y, noisy_y] = noisy_nlsig(x, N, signal_type, 'AWGN', std=std)  # generate curve we want to fit using NN
    # plt.figure(1)
    # plt.plot(x, noisy_y, label='nonlinear function with noise',lw=2)
    # plt.plot(x, y, label='nonlinear function',lw=2)
    # plt.xlabel('x')
    # plt.ylabel('Nonlinear function')
    # plt.legend()
    # plt.xlim([min(x), max(x)])
    # plt.grid(visible=True)
    # plt.savefig('./signal.png', dpi=600)
    # plt.show(block=True)

    net = MLP(1,nodes_num,1)
    net.apply(init_weights)
    # print(net)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    # x = torch.unsqueeze(x, dim=1)
    input_x = torch.unsqueeze(input_x, dim=1)
    # y = torch.unsqueeze(y, dim=1)
    # noisy_y = torch.unsqueeze(noisy_y, dim=1)
    # target_signal = noisy_y # noisy signal / pure signal
    # plt.ion()

    #### start training process ####
    for step in range(epochs):
        x = torch.linspace(-10, 10, N)
        [y, noisy_y] = noisy_nlsig(x, N, signal_type, 'AWGN',std=std)  # use different noise each epoch
        y = torch.unsqueeze(y, dim=1)
        noisy_y = torch.unsqueeze(noisy_y, dim=1)
        target_signal = noisy_y  # noisy signal / pure signal
        x = torch.unsqueeze(x, dim=1)
        x_val = x
        y_val = y

        pred = net(x)
        loss = loss_func(pred.float(), target_signal.float())
        loss_r[step] = loss.data.numpy()
        # loss_r_l[lr,:] = loss_r
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Validation
        net.eval()
        val_output = net(x_val)
        loss_val = loss_func(val_output.float(), y_val.float())
        loss_val_r[step] = loss_val.data.numpy()
        # loss_val_r_l[lr,:] = loss_val_r

        print('epochs: %d, loss: %.8f, val_loss: %.8f' % (step, loss.data.numpy(), loss_val.data.numpy()))

        plt.ion()
        if step % 20 == 0:
            plt.figure(2)
            plt.cla()
            plt.plot(x.data.numpy(), target_signal.data.numpy(),'.',label = 'Signal',lw=2) # target curve
            plt.plot(x.data.numpy(), y.data.numpy(),'black',marker = '^',markevery = 100,lw=2, label = 'True') # cureve without noisy
            plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=1.5, label = 'learning data') # pred result
            # plt.text(.5, 0, 'Loss=%.8f' % loss.data.numpy())
            plt.legend(loc=4)
            plt.grid(visible=True)
            plt.pause(0.01)
            # plt.savefig('./curve/test%lr%i.png' % (lr,step), dpi=600)
        plt.ioff()
# plt.figure(2)
# plt.cla()
# plt.plot(x.data.numpy(), target_signal.data.numpy(),'.',label = 'Signal',lw=2) # target curve
# plt.plot(x.data.numpy(), y.data.numpy(),'black',marker = '^',markevery = 100,lw=2, label = 'True') # cureve without noisy
# # plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=1.5, label = 'learning data ' % (learning_rate_seq[lr])) # pred result
# plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=1.5, label='learning data')  # pred result
# # plt.text(.5, 0, 'Loss=%.8f' % loss.data.numpy())
# plt.legend(loc=4)
# plt.title('Performance of using the optimal parameter')
# plt.xlim([min(x),max(x)])
# plt.grid(visible=True)
# plt.pause(0.01)
# plt.savefig('./curve/test.png', dpi=600)

    plt.figure(3)
    # for l in range(lr_n):
    plt.semilogy(list(range(0, epochs)),loss_r,marker = '*',markevery = 100,lw=2,label='Train learning rate' )
    plt.semilogy(list(range(0,epochs)), loss_val_r,marker = 'o',markevery = 100,label='Validation learning rate')
    plt.legend()
    plt.xlim([0, epochs])
    plt.xlabel('Epoch Numbers')
    plt.ylabel('MSE(dB)')
    plt.grid(visible=True)
    plt.semilogy(list(range(0, epochs)), std ** 2 * np.ones(np.size(loss_r)), 'r', label='Noise variance')
    plt.title('Training loss and Validation loss in the Optimal parameter set')
    plt.savefig('./iterations/step.png', dpi=600)
    plt.show(block=True)


    # for j in range(epochs):
    #     if j % 5 == 0:
    #         plt.figure(4)
    #         plt.plot(list(range(0,j)), loss_r[0:j],label='Train')
    #         plt.plot(list(range(0, j)), loss_val_r[0:j],'black', label='Validation')
    #         plt.plot(list(range(0, j)), std ** 2 * np.ones(np.size(loss_r))[0:j], 'r', label='Noise variance')
    #         plt.xlim([-0.5, epochs])
    #         plt.legend()
    #         plt.xlabel('Epoch Numbers')
    #         plt.ylabel('MSE')
    #         plt.grid(visible=True)
    #         plt.savefig('./iterations/step%d.png' % j, dpi=600)
    #         plt.cla()
            # plt.show(block=True)




