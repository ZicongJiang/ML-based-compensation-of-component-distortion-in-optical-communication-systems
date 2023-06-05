###########################################################################################
### This is the main code for optical communication system simulation
### Author: Zicong Jiang. Time: 2023 May
###########################################################################################
import matplotlib.pyplot as plt
import torch
from format_chose import *
from com_functions import *
from library import *

# torch.manual_seed(0)


def qfunc(x):
    return 0.5 - 0.5 * special.erf(x / torch.sqrt(torch.tensor(2)))


def testbench(N, SNR_dB):
    snr = 10 ** (SNR_dB / 10)
    M = [2, 4, 8]
    # Initial parameters
    SER_tb = torch.zeros([len(M) + 1, N])

    # Calculate the testbench SER for OOK,2PAM,4PAM and 8PAM
    SER_tb[0, :] = qfunc(torch.sqrt(snr))
    for k in range(len(M)):
        SER_tb[k + 1, :] = ((2 * M[k] - 2) / M[k]) * qfunc(np.sqrt((6*snr) / (M[k] ** 2 - 1)))
        #SER_tb[k + 1, :] = ((2 * M[k] - 2) / M[k]) * qfunc(np.sqrt((2/1.25 * snr) ))
    return SER_tb


if __name__ == "__main__":
    N = 1000
    SNR_dB = torch.linspace(0, 25, N)
    symbol_rate = 16  # GHz
    samples_per_symbol = 30
    loopnum = 1

    SER_tb = testbench(N, SNR_dB)

    # 2PAM, 4PAM, 4PAM_MZM:
    # 2PAM = 2PAM + gaussian noise + Quantization noise
    # 4PAM = 4PAM + gaussian noise + Quantization noise
    # 4PAM_MZM = 4PAM + gaussian noise + Quantization noise + MZM
    SER = format_choise('2PAM', N, loopnum, SNR_dB, symbol_rate, samples_per_symbol)

    plt.figure(1)
    plt.cla()
    plt.semilogy(SNR_dB, SER, label='4PAM')
    plt.semilogy(SNR_dB, SER_tb[0, :], label='4PAM_TB')
    plt.ylim([10**(-4), 1])
    plt.xlim(left=0)
    plt.grid(visible=True)
    plt.show(block=True)

    plt.figure(2)
    plt.cla()
    plt.semilogy(SNR_dB, SER_tb[0, :], marker='D', markevery=50, label='OOK')
    plt.semilogy(SNR_dB, SER_tb[1, :], marker='v', markevery=50, label='2PAM')
    plt.semilogy(SNR_dB, SER_tb[2, :], marker='*', markevery=50, label='4PAM')
    plt.semilogy(SNR_dB, SER_tb[3, :], marker='o', markevery=50, label='8PAM')
    plt.legend()
    plt.xlabel('SNR(dB)')
    plt.ylabel('SER')
    plt.title('SER testbench for OOK, 2PAM, 4PAM and 8PAM')
    plt.ylim([1.0000e-07, 1])
    plt.xlim(left=0)
    plt.grid(visible=True)
    plt.show(block=True)
