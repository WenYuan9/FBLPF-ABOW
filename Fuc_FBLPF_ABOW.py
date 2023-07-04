from Fuc import RMS,LL,CC, pd,func_transformer,SCA,LowPass, \
    plt, Remove_DWT,copy,np,Determine_Pattern,MyFilterBank,pywt

def Remove(Data,fs,):
    # ------------------------------------------------------------------------------------------------
    u_d = LowPass(Data, 12, fs, 4)
    Peak_index_High = []
    theda =RMS(u_d)/RMS(Data)* (((np.median(abs(u_d))) / 0.6745) ** 1) * ((2 * (np.log(len(u_d)))) ** 0.5)
    for temp_i in np.arange(1, len(u_d) - 1):
        if u_d[temp_i] > u_d[temp_i - 1] and u_d[temp_i] > u_d[temp_i + 1] and u_d[temp_i] > theda:
            Peak_index_High.append(temp_i)
    # ------------------------------------------------------------------------------------------------
    PatternSignalList = []
    StartList = []
    FinalList = []
    for temp_peak in Peak_index_High:
        start = np.max([int(temp_peak - (0.6 * 1/6) * fs),0])
        final = np.min([int(temp_peak + (0.6* 5/6) * fs),len(Data)])

        PatternSignalList.append(Data[start:final])
        StartList.append(start)
        FinalList.append(final)
    # ------------------------------------------------------------------------------------------------
    Pattern_Index = Determine_Pattern(PatternSignalList)
    Pattern_Signal = PatternSignalList[Pattern_Index]
    # ------------------------------------------------------------------------------------------------
    def demo_func(Par_Init):
        # ------------------------------------------------------------------------------------
        myfilterbank = MyFilterBank()
        myfilterbank.Get_a(Par_Init)
        myWavelet = pywt.Wavelet(name="myWavelet", filter_bank=myfilterbank.filter_bank)
        # ------------------------------------------------------------------------------------
        # Loss_1:相似性损失
        wave = pywt.wavedec(Pattern_Signal, wavelet=myWavelet, level=4)
        rec_signal = pywt.waverec(np.multiply(wave, [1,]+[0]*4).tolist(), wavelet=myWavelet)
        min_1=np.min([len(Pattern_Signal),len(rec_signal)])
        # Loss_1 = np.array(Pattern_Signal[:min_1]) - np.array(rec_signal[:min_1])
        # Loss_1 = np.average(abs(Loss_1) ** 2)
        Loss_1=CC(Pattern_Signal[:min_1] , np.array(rec_signal[:min_1]))
        # ------------------------------------------------------------------------------------
        return 1-Loss_1
    # ------------------------------------------------------------------------------------------------
    sca=SCA(func=demo_func, max_iter=15)
    # pso = PSO(func=demo_func, max_iter=15, dim=1,)
    # fitness = pso.run(Path="C:/Users/WenYuan/Desktop/Wavelt/SIMU/Fig/High/PSO/"+str(Index)+".svg")
    fitness = sca.run()
    # ------------------------------------------------------------------------------------------------
    myfilterbank = MyFilterBank()
    myfilterbank.Get_a(sca.gbest_x)
    myWavelet = pywt.Wavelet(name="myWavelet", filter_bank=myfilterbank.filter_bank)
    # ------------------------------------------------------------------------------------------------
    temp_noise_data_Copy = copy.deepcopy(Data)
    # ------------------------------------------------------------------------------------------------
    for temp_start, temp_final in zip(StartList, FinalList):
        # ------------------------------------------------------------------------------------------------
        real_L=LL(Data[temp_start:temp_final],0.85,myWavelet)
        temp_filter_data = Remove_DWT(Data[temp_start:temp_final],real_L, myWavelet,Flag=False)
        min_len = np.min([len(temp_noise_data_Copy[temp_start:temp_final]), len(temp_filter_data)])
        temp_noise_data_Copy[temp_start:temp_final][:min_len] = temp_filter_data[:min_len]
    return temp_noise_data_Copy