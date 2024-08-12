
import math
import numpy as np
from math import e
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter

def normal_distribution(x,mu,sig):
    sig = sig*sig
    ans = 1 / (sig* (2*math.pi)**0.5) * math.exp(- (x-mu)**2 / (2*sig**2))
    return ans

class Time_Frequency():
    def __init__(self):
        self.fs = 1000 # sampling rate取樣率
        self.ts = 1/self.fs # sampling interval 取樣區間 (1/Fs)
        self.num_channel = 8

    #---------------------------------------------------------------------#
    #                                  FFT
    #---------------------------------------------------------------------#
    def fft(self, eeg_data):
        """
        FFT轉換，輸入資料一維

        Parameters
        ----------------------------------------
        `eeg_data` : with shape = (samples, ) which 
                   n = number of sample points

        Returns
        ----------------------------------------
        `t` : number of time points with shape = (samples, )

        `freq1` : number of frequency points 

        `Y1` : FFT transform with shape = (freq, )

        Examples
        ----------------------------------------
        >>> tfa = Time_Frequency()
        >>> tfa.data_check(eeg_data)
        """       

        y = eeg_data

        t = np.arange(0, len(y)/self.fs, self.ts)  # time vector,這裡Ts也是步長

        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n/self.fs
        freq = k/T  # two sides frequency range
        freq1 = freq[range(int(n/20))]  # one side frequency range

        # np.fft.fft(y) # 未歸一化
        Y = np.fft.fft(y)/n   # fft computing and normalization 歸一化
        Y1 = Y[range(int(n/20))]
        return t, freq1, Y1

    def plot_fft(self, eeg_data, png_path, is_smooth=False, ):
        """
        Parameters
        ----------------------------------------
        `eeg_data` : with shape = (ch, samples) which 
                     ch is the number of channel,
                     samples is the number of sample points

        `is_smooth` : True : smooth the FFT data
                      False : original FFT data

        `png_path` : path for save figure


        Examples
        ----------------------------------------
        >>> tfa = Time_Frequency()
        >>> tfa.plot_fft(eeg_data)
        """        

        label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

        fig, ax = plt.subplots(self.num_channel, 2, figsize=(14,6))


        nperseg = 4 * self.fs
        from scipy import signal

        for channel in range(self.num_channel):
            y = eeg_data[channel]
            t, freq, Y = self.fft(y)
            if is_smooth:
                Y = savgol_filter(abs(Y), len(Y)//100, 3) # smooth the data, window size = len(Y)//100, polynomial order 3
                title_annotation = ' (Smoothed)'
            else:
                title_annotation = ''

      
                        
            # #17395C
            # plot raw data
            ax[channel, 0].plot(t, y, linewidth=0.5, color='#17395C', alpha=0.9)
            ax[channel, 0].set_ylabel(label_list[channel], fontsize=12, rotation=0) # 'Amplitude'
            ax[channel, 0].yaxis.set_label_coords(-.2, .4)
            # plot spectrum
            ax[channel, 1].plot(freq, abs(Y), linewidth=0.5, color='gray', alpha=0.9)
            ax[channel, 1].set_ylabel(label_list[channel], fontsize=12, rotation=0) # '|Y(freq)|'
            ax[channel, 1].yaxis.set_label_coords(1.1, .4)


            # welch's method psd
            #   https://raphaelvallat.com/bandpower.html
            f, psd = signal.welch(eeg_data[channel], self.fs, nperseg=nperseg)
            ratio = np.max(abs(Y)) // np.max(psd)
            ax[channel, 1].plot(f, psd*ratio, linewidth=1.0, color='k', alpha=0.9)
            ax[channel, 1].set_xlim([0, 50])          


            # remove x label except the bottom plot
            if (channel + 1 != self.num_channel):
                ax[channel, 0].axes.xaxis.set_ticklabels([])
                ax[channel, 1].axes.xaxis.set_ticklabels([])
            else:
                ax[channel, 0].set_xlabel('Time', fontsize=10)
                ax[channel, 1].set_xlabel('Freq (Hz)', fontsize=10)
            
            # mark the range of alpha band (8 - 12 Hz)
            rect = patches.Rectangle((8,0),
                            width=4,
                            height=np.max(abs(Y)),
                            facecolor = 'orange',
                            fill = True,
                            alpha = 0.4)             
            ax[channel, 1].add_patch(rect)

        ax[0, 0].set_title('Raw EEG', fontsize=12)
        ax[0, 1].set_title('FFT / Welch' + title_annotation, fontsize=12)

        plt.savefig(png_path, dpi=500)





    def data_check(self, eeg_data, con=0.7):
        """
        檢查腦波資料經過FFT轉換後的頻譜能量有沒有集中在alpha band，如果超過60分就算合格

        Parameters
        ----------------------------------------
        `eeg_data` : eeg data with shape = (ch, samples) which
                     ch is the number of channel,
                     samples is the number of sample points  


        Examples
        ----------------------------------------
        >>> tfa = Time_Frequency()
        >>> tfa.data_check(eeg_data)
        """ 
        
        # con = 0.3 #越小越集中，標準越嚴格(>=0.4)
        num_channel = 8

        print("channel  |   E_score")
        for channel in range(num_channel):
            y = eeg_data[channel]
            t, freq, Y = self.fft(y)

            E_good_data = E_con_data = E_bad_data = E_con_data_M = 0
            eg = 0        
            max_i = 0
            
            Y = abs(Y)        

            for i in range(len(freq)):         
                if (freq[i] > 8 and freq[i] < 12):                
                    E_good_data = E_good_data + Y[i]**2
                    if (Y[max_i]**2 < Y[i]**2):
                        max_i = i
                    eg = eg+1            
                else:
                    E_bad_data = E_bad_data + Y[i]**2              
            Er_out = E_good_data/(E_good_data + E_bad_data) 

            E_good_avg = E_good_data/eg
            m01 = Y[max_i]**2/normal_distribution(freq[max_i],freq[max_i],con)

            # for i in range(len(freq)):             
            #     if (freq[i] > 8 and freq[i] < 12):
            #         E_con_data = E_con_data + m01*normal_distribution(freq[i],10,con) - Y[i]**2
            #         E_con_data_M = E_con_data_M + m01*normal_distribution(freq[i],10,con)
            #         #常態分佈作為標準函數
            # Er_in = E_con_data/E_con_data_M*100
            # E_score = Er_in * Er_out 
            
            for i in range(len(freq)):             
                if (freq[i] > 8 and freq[i] < 12):
                    E_con_data = E_con_data + m01*normal_distribution(freq[i],freq[max_i],con) - Y[i]**2
                    E_con_data_M = E_con_data_M + m01*normal_distribution(freq[i],freq[max_i],con)
                    #常態分佈作為標準函數
            Er_in = E_con_data / E_con_data_M * 100
            Er_shift = 10 / (10 + abs(10-freq[max_i]))
            E_score = Er_in * Er_out * Er_shift *2.5
            
            if E_score <= 60: color = "91" # red
            else : color = "92" # green
            print("   \033[{:s}m{}     |  {:6.2f}\033[0m %".format(color, channel+1, E_score)) # 

    #---------------------------------------------------------------------#
    #                               Wavelet
    #---------------------------------------------------------------------#
    def plot_wavelet(self, eeg_raw, mu_low = 8, mu_high = 12): 
        """
        檢查腦波資料經過FFT轉換後的頻譜能量有沒有集中在alpha band，如果超過60分就算合格

        Parameters
        ----------------------------------------
        `eeg_data` : eeg data with shape = (ch, samples) which
                     ch is the number of channel,
                     samples is the number of sample points
        `mu_low`, `mu_high` : frequency range start from mu_low to mu_high. e.g. 8 ~ 12 Hz


        Examples
        ----------------------------------------
        >>> tfa = Time_Frequency()
        >>> tfa.data_check(eeg_data)
        """ 

        import seaborn as sns

        freq_gap = mu_high - mu_low + 1
        wavelet_eeg = self.wavelet(np.array(eeg_raw), mu_low=mu_low, mu_high=mu_high)

        channel_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
        yticklabels_freq = [str(i) for i in range(mu_low, mu_high+1)] # frequency range
        xticklabels_time = [str(i/1000) for i in range(0, len(wavelet_eeg[0]), 500)] # timestamp, interval 0.5 s
        fig, ax = plt.subplots(self.num_channel, 1, figsize=(15,16))

        for i in range(self.num_channel):
            sub_wavelet_eeg = wavelet_eeg[i*freq_gap:(i+1)*freq_gap] # get wavelet of each channel 
            sns.heatmap(data=sub_wavelet_eeg, 
                        ax=ax[i],
                        cmap=plt.get_cmap('jet'),
                        cbar=False,
                        yticklabels=channel_list[i],
                        vmax=np.max(wavelet_eeg),
                        vmin=np.min(wavelet_eeg))

            yticks = np.linspace(1, freq_gap, freq_gap)
            ax[i].set_yticks(yticks) # set the number of yticks to display
            ax[i].set_yticklabels(yticklabels_freq, rotation=0) # set the text of yticks
            ax[i].set_ylabel(channel_list[i], rotation=0, fontsize=14) # ylabel (name of channel)
            ax[i].yaxis.set_label_coords(-0.05, .4) # set the label position

            xticks = [i for i in range(0, len(wavelet_eeg[0]), 500)]
            ax[i].set_xticks(xticks) # set the number of xticks to display

            # remove x label except the bottom plot
            if (i + 1 != self.num_channel):
                ax[i].axes.xaxis.set_ticklabels([]) 
            else:  
                ax[i].set_xticklabels(xticklabels_time, rotation=0) # set the text of xticks
                ax[i].set_xlabel('Time [sec]')

        ax[0].set_title('Wavelet', fontsize=14)
        plt.show()

    def MorletWavelet(self, fc):
        F_RATIO = 7
        Zalpha2 = 3.3
        sigma_f = fc / F_RATIO
        sigma_t = 1 / (2 * math.pi * sigma_f)
        A = 1 / math.sqrt(sigma_t * math.sqrt(math.pi))
        max_t = math.ceil(Zalpha2 * sigma_t)

        t = np.arange(-max_t, max_t+1, 1)
        v1 = 1 / (-2 * sigma_t**2)
        v2 = 2j * math.pi * fc
        MV = A * e**(t*(t*v1 + v2))
        return MV

    def tfa_morlet(self, td, fmin, fmax, fstep):  # (data,1000,4,14,1)
        lens = len(td)
        TFmap1 = np.zeros((1, lens))

        for x in np.arange(fmin, fmax + 1, fstep):
            MW = self.MorletWavelet(x/self.fs)
            cr = np.convolve(td, MW, 'same')
            cr2 = abs(cr).reshape(1, lens)
            TFmap1 = np.append(TFmap1, cr2, axis=0)
        TFmap = np.delete(TFmap1, 0, axis=0)
        return TFmap

    def wavelet(self, cut_data, mu_low=8, mu_high=12):
        """
        小波轉換

        Parameters
        ----------------------------------------
        `cut_data` : eeg data with 
                        shape = (n, ch, samples) for single trial or
                        shape = (ch, samples) for multi trials, which
                     n is number of trials, 
                     ch is number of channels, and,
                     samples is number of sample points

        `mu_low`, `mu_hight` : range of selected frequence ( 8 - 12 = alpah band)

        Returns
        ----------------------------------------
        `EEG_morlet_all[0]` : wavelet transform for single trial

        `EEG_morelet_avg` : wavelet transform for multi trials
        with shape are = ((mu_hight - mu_low + 1)*8, samples)

        Examples
        ----------------------------------------
        >>> tfa = Time_Frequency()
        >>> right_wavelet = tfa.wavelet(right_eeg, mu_low = 8, mu_high = 12)
        """ 

        print("shape of input_data = {}".format(cut_data.shape))
        EEG_morlet_all = []
        if cut_data.ndim == 2:            
            
            ch1 = self.tfa_morlet(cut_data[0][:], mu_low, mu_high, 1)
            ch2 = self.tfa_morlet(cut_data[1][:], mu_low, mu_high, 1)
            ch3 = self.tfa_morlet(cut_data[2][:], mu_low, mu_high, 1)
            ch4 = self.tfa_morlet(cut_data[3][:], mu_low, mu_high, 1)
            ch5 = self.tfa_morlet(cut_data[4][:], mu_low, mu_high, 1)
            ch6 = self.tfa_morlet(cut_data[5][:], mu_low, mu_high, 1)
            ch7 = self.tfa_morlet(cut_data[6][:], mu_low, mu_high, 1)
            ch8 = self.tfa_morlet(cut_data[7][:], mu_low, mu_high, 1)

            EEG_morlet = np.concatenate((ch1, ch2), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch3), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch4), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch5), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch6), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch7), 0)
            EEG_morlet = np.concatenate((EEG_morlet, ch8), 0)

            EEG_morlet_all.append(EEG_morlet)
            EEG_morlet_all = np.array(EEG_morlet_all)
            print("shape of EEG_morlet_all = {}".format(EEG_morlet_all[0].shape))
            return EEG_morlet_all[0] 

        elif cut_data.ndim == 3:

            for i in range(len(cut_data)):      
                ch1 = self.tfa_morlet(cut_data[i][0][:], mu_low, mu_high, 1)
                ch2 = self.tfa_morlet(cut_data[i][1][:], mu_low, mu_high, 1)
                ch3 = self.tfa_morlet(cut_data[i][2][:], mu_low, mu_high, 1)
                ch4 = self.tfa_morlet(cut_data[i][3][:], mu_low, mu_high, 1)
                ch5 = self.tfa_morlet(cut_data[i][4][:], mu_low, mu_high, 1)
                ch6 = self.tfa_morlet(cut_data[i][5][:], mu_low, mu_high, 1)
                ch7 = self.tfa_morlet(cut_data[i][6][:], mu_low, mu_high, 1)
                ch8 = self.tfa_morlet(cut_data[i][7][:], mu_low, mu_high, 1)

                EEG_morlet = np.concatenate((ch1, ch2), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch3), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch4), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch5), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch6), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch7), 0)
                EEG_morlet = np.concatenate((EEG_morlet, ch8), 0)

                EEG_morlet_all.append(EEG_morlet)
            
            EEG_morelet_avg = np.average(EEG_morlet_all, axis=0) # average all column
            print("shape of EEG_morelet_avg = {}".format(EEG_morelet_avg.shape))
            return EEG_morelet_avg              
        else:
            print("Require ndim = 2 or 3, but get ndim = {}".format(cut_data.ndim))
    