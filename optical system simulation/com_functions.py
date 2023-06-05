from math import *

import torch

import format_chose
from library import *
from format_chose import *


def pulse_shape(N, Rs, L, symbolsPAM):
    Ts = 1 / Rs
    Tsa = Ts / L
    time = torch.arange(-Ts / 2, Ts / 2, Tsa)

    # Gaussian pulse
    Tfwhm = 1 / (6 * Rs)
    T0 = Tfwhm / (2 * torch.sqrt(torch.log(torch.tensor(2))))
    Gaussian = torch.exp((-1 / 2) * torch.square(time / T0))

    # Gaussian pulse shaping
    waveformPAM = symbolsPAM.repeat([L, 1])
    waveformPAMup = torch.reshape(torch.t(waveformPAM), (-1,))

    index = 0
    waveGaussianPAM = torch.zeros(N, L)
    for i in range(1, len(waveformPAMup), L):
        waveGaussianPAM[index, :] = waveformPAMup[i - 1:(i + L - 1)] * Gaussian
        index = index + 1

    waveGaussianPAM = torch.reshape(waveGaussianPAM, (-1,))

    return waveGaussianPAM


def format_choise(format, N, loopnum, dB, symbol_rate, samples,x):
    snr = 10 ** (dB / 10)
    if format == '2PAM':
        SER = format_chose.PAM_2(N, loopnum, snr, symbol_rate, samples)
    elif format == '4PAM':
        SER = format_chose.PAM_4(N, loopnum, snr, symbol_rate, samples)
    elif format == '4PAM_MZM':
        SER = format_chose.PAM_4_MZM(N, loopnum, snr, symbol_rate, samples)
    elif format == '8PAM':
        SER = format_chose.PAM_8(N, loopnum, snr, symbol_rate, samples)
    elif format == 'train':
        SER = format_chose.PAM_4_MZM_train(x, N, symbol_rate, samples)

    return SER


def Quantnoise_Tx(Tx_signal, Eav):
    ENOB = 6
    P = Eav
    variance = 3 * P * 10 ** (-(6.02 * ENOB + 1.76) / 10)
    ENOB_noise = torch.sqrt(variance) * torch.randn(len(Tx_signal))
    Tx_signal_noise = Tx_signal + ENOB_noise
    return Tx_signal_noise


def Quantnoise_Rx(Rx_signal, Eav):
    ENOB = 6
    P = Eav
    variance = 3 * P * 10 ** (-(6.02 * ENOB + 1.76) / 10)
    ENOB_noise = torch.sqrt(variance) * torch.randn(len(Rx_signal))
    Tx_signal_noise = Rx_signal + ENOB_noise
    return Tx_signal_noise


def MZM(x, Vpi, Vdc, Vpp):
    Vi = Vdc + Vpp * x
    Eout = torch.square((torch.cos((pi / 2) * (Vi / Vpi))))
    return Eout
