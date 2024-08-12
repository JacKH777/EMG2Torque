#%%
# 解碼收到的腦波資料 EEG.txt，並做濾波
# 畫 FFT & Wavelet transform

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# import custom libraries
from eeg_decoder import Decoder, get_latest_date, findall_datadate_dirs
from eeg_tfa import Time_Frequency
# %config InlineBackend.figure_format = 'retina'
# %config InlineBackend.figure_format = 'svg'

def show_leadOff(eeg_data):
    # 每個channel中的leadOff只要出現一個1就視為斷線
    if eeg_data.shape[0] <= 10:
        print("\nNo leadOff")
        return
    lead_off = [str(int(x)) for x in eeg_data[10]]

    # 補0，leadOff長度有可能不是8
    # e.g. 100, 1110, 0, ...
    for i, x in enumerate(lead_off):
        if len(x) < 8:
            num_padding = 8 - len(x)
            lead_off[i] = '0'*num_padding + lead_off[i]

    # string turn into list
    # e.g. 1100010 -> [1, 1, 0, 0, 0, 1, 0]
    lead_off = np.array([list(x) for x in lead_off], dtype=int).T

    # if there more than 1 consider as disconnected
    connect_state = np.sum(lead_off, axis=1)
    color = [91 if x else 92 for x in connect_state]
    # 91 : red, 92 green

    # print layout
    # ╭─ EEG Electrode Connection ─╮
    # │   F3                  F4   │
    # │   C3        Cz        C4   │
    # │   P3        Pz        P4   │
    # ╰─ connected ─ disconnected ─╯    
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # u'\u2502' = "│" , u'\u256D' = "╭", u'\u2500' = "─", u'\u256E' = "╮", u'\u2570' = "╰", u'\u256F' = "	╯"
    print('\n')
    print(u'\u256D' + u'\u2500' + ' EEG Electrode Connection ' + u'\u2500' +  u'\u256E')
    print(u'\u2502' + f"\033[{color[0]}m{'F3':^9}\033[0m {'':^9} \033[{color[1]}m{'F4':^8}\033[0m" + u'\u2502')
    print(u'\u2502' + f"\033[{color[2]}m{'C3':^9}\033[0m \033[{color[3]}m{'Cz':^9}\033[0m \033[{color[4]}m{'C4':^8}\033[0m" + u'\u2502') 
    print(u'\u2502' + f"\033[{color[5]}m{'P3':^9}\033[0m \033[{color[6]}m{'Pz':^9}\033[0m \033[{color[7]}m{'P4':^8}\033[0m" + u'\u2502')
    print(u'\u2570' + u'\u2500' + "\033[92m connected \033[0m" + u'\u2500' + "\033[91m disconnected \033[0m" + u'\u2500' + u'\u256F')


if __name__ == '__main__':
    decoder = Decoder()
    # =========================================================================== #
    # [Parameter setting]                       
    # =========================================================================== #    
    show_fft     = 1
    show_wavelet = 0

    fs = 1000

    EXP_DIR   = './exp'
    data_date = get_latest_date(EXP_DIR) 
    # data_date = '2024_07_05_1123'

    # =========================================================================== #
    # [Decode]                       
    # =========================================================================== #   


    eeg_txt_path      = f'{EXP_DIR}/{data_date}/1/EEG.txt'
    decoded_file_path = f'{EXP_DIR}/{data_date}/1/1.txt'

    if (os.path.exists(decoded_file_path)):
        # 讀取解碼後的EEG資料 1.txt，畫圖
        eeg_data = decoder.read_decoded(decoded_file_path) 
        decoder.plot_eeg(eeg_data, 
                        png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/EEG.png')
    else:
        # 解碼16進制EEG.txt資料至10進制 1.txt 和 1.npy
        eeg_data = decoder.decode_to_txt(eeg_txt_path = eeg_txt_path, 
                                        return_data = True,
                                        decode_to_npy = True) 


    # =========================================================================== #
    # [show info]                       
    # =========================================================================== # 
    #   
    # 捨棄前一秒的資料以及最後一秒的資料，
    # 因為濾波後的訊號在這兩段時間會有異常波形
    start    = 1 * fs 
    end      = eeg_data.shape[0] - (1 * fs)
    eeg_data = eeg_data[start:end].T

    # 顯示腦波電極連接狀況
    show_leadOff(eeg_data)

    # 畫 FFT or wavelet
    tfa = Time_Frequency()
    if show_fft:
        print("\nPlot FFT")        
        tfa.plot_fft(eeg_data, 
                    png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/FFT.png', 
                    is_smooth=False)
    if show_wavelet:
        print("\nPlot Wavelet transform")   
        # 頻率範圍 mu_low ~ mu_high Hz, step = 1
        tfa.plot_wavelet(eeg_data, mu_low=8, mu_high=14)
    plt.show()


 
#

# %%
