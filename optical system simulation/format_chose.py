import random

import numpy as np
import torch
from com_functions import *
from library import *

def PAM_8(N, loopnum, snr, symbol_rate, samples):
    Rs = symbol_rate
    L = samples
    SER_PAM = torch.zeros(len(snr))
    SER_PAM_sum = torch.zeros(len(snr))

    for n in range(loopnum):
        x = torch.tensor((np.random.randint(8,size=N)).astype(float))
        # pulse shaping for input symbol
        x_gauss = pulse_shape(N, Rs, L, x)
        Eav = torch.mean(torch.square(x))

        for i in range(len(snr)):
            N_0 = Eav/snr[i]/2
            ni = np.sqrt(N_0)*np.random.randn(len(x_gauss))
            yR = x_gauss+ni

            # sample before SER detection
            samplePerSymbol = int(len(x_gauss)/N)
            Etx_downsampled = yR[int((samplePerSymbol/2)):-1:samplePerSymbol]
            y_detect = torch.zeros(len(Etx_downsampled))
            for k in range(len(snr)):
                if Etx_downsampled[k] < 1/2:
                    y_detect[k] = 0
                elif Etx_downsampled[k] < 3/2:
                    y_detect[k] = 1
                elif Etx_downsampled[k] < 5/2:
                    y_detect[k] = 2
                elif Etx_downsampled[k] < 7/2:
                    y_detect[k] = 3
                elif Etx_downsampled[k] < 9/2:
                    y_detect[k] = 4
                elif Etx_downsampled[k] < 11/2:
                    y_detect[k] = 5
                elif Etx_downsampled[k] < 13/2:
                    y_detect[k] = 6
                else:
                    y_detect[k] = 7

            bit_R = torch.sum((x == y_detect) == 0)
            SER_PAM[i] = bit_R/N
        SER_PAM_sum = SER_PAM+SER_PAM_sum
    SER_PAM = SER_PAM_sum/loopnum

    return SER_PAM


def PAM_2(N, loopnum, snr, symbol_rate, samples):
    Rs = symbol_rate
    L = samples
    # symbol_2pam = [0,1]
    SER_2PAM = torch.zeros(len(snr))
    SER_2PAM_sum = torch.zeros(len(snr))

    for n in range(loopnum):
        x = (torch.rand(N) < 0.5).float()
        # pulse shaping for input symbol
        x_gauss = pulse_shape(N, Rs, L, x)
        Eav = torch.mean(torch.square(x))
        x_gauss = Quantnoise_Tx(x_gauss, Eav)


        for i in range(len(snr)):
            N_0 = Eav/snr[i]/2
            ni = np.sqrt(N_0)*np.random.randn(len(x_gauss))
            yR = x_gauss+ni

            # Quantization noise
            # calculate power for ENOB receiver
            n_0 = np.sqrt(N_0)*np.random.randn(len(x))
            x_noise = x+n_0
            yR = Quantnoise_Rx(yR, torch.mean(np.square(abs(x_noise))))

            # sample before SER detection
            samplePerSymbol = int(len(x_gauss)/N)
            Etx_downsampled = yR[int((samplePerSymbol/2)):-1:samplePerSymbol]
            y_detect = torch.zeros(len(Etx_downsampled))
            for k in range(len(snr)):
                if Etx_downsampled[k] < 1/2:
                    y_detect[k] = 0
                else:
                    y_detect[k] = 1
            bit_R = torch.sum((x == y_detect) == 0)
            SER_2PAM[i] = bit_R/N
        SER_2PAM_sum = SER_2PAM+SER_2PAM_sum
    SER_2PAM = SER_2PAM_sum/loopnum

    return SER_2PAM



def PAM_4(N, loopnum, snr, symbol_rate, samples):
    Rs = symbol_rate
    L = samples
    SER_PAM = torch.zeros(len(snr))
    SER_PAM_sum = torch.zeros(len(snr))

    for n in range(loopnum):
        x = torch.tensor((np.random.randint(4,size=N)).astype(float))
        # pulse shaping for input symbol
        x_gauss = pulse_shape(N, Rs, L, x)
        # Eav = torch.mean(torch.square(x))
        Eav = torch.tensor(1.25)
        x_gauss = Quantnoise_Tx(x_gauss, Eav)


        for i in range(len(snr)):
            N_0 = Eav/snr[i]/2
            ni = np.sqrt(N_0)*np.random.randn(len(x_gauss))
            yR = x_gauss+ni

            # Quantization noise
            # calculate power for ENOB receiver
            n_0 = np.sqrt(N_0)*np.random.randn(len(x))
            x_noise = x+n_0
            yR = Quantnoise_Rx(yR, torch.mean(np.square(abs(x_noise))))

            # sample before SER detection
            samplePerSymbol = int(len(x_gauss)/N)
            Etx_downsampled = yR[int((samplePerSymbol/2)):-1:samplePerSymbol]
            y_detect = torch.zeros(len(Etx_downsampled))
            for k in range(len(snr)):
                if Etx_downsampled[k] < 1/2:
                    y_detect[k] = 0
                elif Etx_downsampled[k] < 3/2:
                    y_detect[k] = 1
                elif Etx_downsampled[k] < 5/2:
                    y_detect[k] = 2
                else:
                    y_detect[k] = 3

            bit_R = torch.sum((x == y_detect) == 0)
            SER_PAM[i] = bit_R/N
        SER_PAM_sum = SER_PAM+SER_PAM_sum
    SER_PAM = SER_PAM_sum/loopnum

    return SER_PAM


def PAM_4_MZM(N, loopnum, snr, symbol_rate, samples):
    Vpi = 1
    Vdc = 1
    Vpp = 1/3
    Rs = symbol_rate
    L = samples
    SER_PAM = torch.zeros(len(snr))
    SER_PAM_sum = torch.zeros(len(snr))

    # x = torch.tensor((np.random.randint(4, size=N)).astype(float))
    # Eout = MZM(x,Vpi,Vdc,Vpp)
    # Eav = torch.mean(torch.square(Eout))
    Eav = 7/18

    for n in range(loopnum):
        x = torch.tensor((np.random.randint(4,size=N)).astype(float))

        # pulse shaping for input symbol
        x_gauss = pulse_shape(N, Rs, L, x)
        x_gauss = Quantnoise_Tx(x_gauss, Eav)

        Eout = MZM(x_gauss,Vpi,Vdc, Vpp)

        for i in range(len(snr)):
            N_0 = Eav/snr[i]/2
            ni = np.sqrt(N_0)*np.random.randn(len(Eout))
            yR = Eout+ni

            # Quantization noise
            # calculate power for ENOB receiver
            n_0 = np.sqrt(N_0)*np.random.randn(len(x))
            x_noise = x/3+n_0
            yR = Quantnoise_Rx(yR, torch.mean(np.square(abs(x_noise))))

            # sample before SER detection
            samplePerSymbol = int(len(x_gauss)/N)
            Etx_downsampled = yR[int((samplePerSymbol/2)):-1:samplePerSymbol]
            y_detect = torch.zeros(len(Etx_downsampled))
            for k in range(len(snr)):
                if Etx_downsampled[k] < 1/6:
                    y_detect[k] = 0
                elif Etx_downsampled[k] < 3/6:
                    y_detect[k] = 1
                elif Etx_downsampled[k] < 5/6:
                    y_detect[k] = 2
                else:
                    y_detect[k] = 3

            bit_R = torch.sum((x == y_detect) == 0)
            SER_PAM[i] = bit_R/N
        SER_PAM_sum = SER_PAM+SER_PAM_sum
    SER_PAM = SER_PAM_sum/loopnum

    return SER_PAM


def PAM_4_MZM_train(x, N, symbol_rate, samples):
    Vpi = 1
    Vdc = 1
    Vpp = 1/3
    Rs = symbol_rate
    L = samples
    Eav = torch.mean(torch.square(x))

    # pulse shaping for input symbol
    x_gauss = pulse_shape(N, Rs, L, x)
    x_gauss = Quantnoise_Tx(x_gauss, Eav)

    yR = MZM(x_gauss,Vpi,Vdc, Vpp)

    # sample before SER detection
    samplePerSymbol = int(len(x_gauss)/N)
    Etx_downsampled = yR[int((samplePerSymbol/2)):-1:samplePerSymbol]

    return Etx_downsampled