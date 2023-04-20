import numpy as np
import torch
from torch import nn
from main import noisy_nlsig, init_weights


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, in_fea, n_hidden, out_fea):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_fea, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.active = nn.Sigmoid()
        self.out = nn.Linear(n_hidden, out_features=out_fea)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x = self.active(self.hidden1(x))
        x = self.active(self.hidden2(x))
        x = self.active(self.hidden3(x))
        x = self.out(x)
        return x


def net_optimizor(std, N, epochs, learning_rate_range, nodes_num_range):
    signal_type = 'GMM'
    loss_r = np.zeros([len(learning_rate_range), len(nodes_num_range),epochs])
    loss_r_val = np.zeros([len(learning_rate_range), len(nodes_num_range),epochs])
    loss_r_op = np.zeros([len(learning_rate_range), len(nodes_num_range)])
    loss_r_val_op = np.zeros([len(learning_rate_range), len(nodes_num_range)])

    for lr in range(len(learning_rate_range)):
        print('############### learning_rate = %d ################' % (learning_rate_range[lr]))
        for nodes_n in range(len(nodes_num_range)):
            print('############### nodes_num = %d ################' % (nodes_num_range[nodes_n]))
            nodes_num = nodes_num_range[nodes_n]
            net = MLP(1, nodes_num, 1)
            net.apply(init_weights)
            # print(net)

            learning_rate = learning_rate_range[lr]
            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            #### start training process ####
            for step in range(epochs):
                x = torch.linspace(-10, 10, N)
                [y, noisy_y] = noisy_nlsig(x, N, signal_type, 'AWGN', std=std)  # use different noise each epoch
                y = torch.unsqueeze(y, dim=1)
                noisy_y = torch.unsqueeze(noisy_y, dim=1)
                target_signal = noisy_y  # noisy signal / pure signal
                x = torch.unsqueeze(x, dim=1)

                x_val = x
                y_val = y

                pred = net(x)
                loss = loss_func(pred.float(), target_signal.float())
                loss_r[lr, nodes_n, step] = loss.data.numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## Validation
                net.eval()
                val_output = net(x_val)
                loss_val = loss_func(val_output.float(), y_val.float())
                loss_r_val[lr, nodes_n, step] = loss_val.data.numpy()

                print('lr:%d, nn:%d, epochs: %d, loss: %.8f, val_loss: %.8f' % (lr, nodes_n, step, loss.data.numpy(), loss_val.data.numpy()))

            loss_r_op[lr, nodes_n] = np.mean(loss_r[lr, nodes_n, 500:1000])
            loss_r_val_op[lr, nodes_n] = np.mean(loss_r_val[lr, nodes_n, 500:1000])

    [row_I_train, col_I_train] = np.where(loss_r_op == np.min(loss_r_op))
    learning_rate_train_opt = learning_rate_range[row_I_train]
    nodes_num_train_opt = nodes_num_range[col_I_train[0]]

    [row_I_val, col_I_val] = np.where(loss_r_val_op == np.min(loss_r_val_op))
    learning_rate_val_opt = learning_rate_range[row_I_val]
    nodes_num_val_opt = nodes_num_range[col_I_val[0]]

    return learning_rate_train_opt, nodes_num_train_opt, learning_rate_val_opt, nodes_num_val_opt
