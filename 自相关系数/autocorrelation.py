from obspy import read
import numpy as np


def Autocorrelation(miniseed_path, p_time, s_time, sec_to_decay=5, sample=5):
    '''
        自相关系数计算，参考《基于Bagging集成学习算法的地震事件性质识别分类》3.3章节

        args:
            miniseed_path: seed文件路径
            p_time: P波到达绝对时间 (obspy.core.utcdatetime.UTCDateTime)
            s_time: ...
            sec_to_decay: 经验系数：非天然震动事件Ｐ波能量衰减主要在5s左右完成。 default=5
            sample: 采样长度，单位为秒

        return: boolean, problem, map
            状态(bool): 算法计算状态，True表示完成计算未出现问题，False为计算过程中发现问题
            异常(string): 具体异常
            计算结果(set)：对应在BHE BHN BHZ上的计算结果
    '''
    seed = read(miniseed_path)
    
    # 检查三个分量的波形数据是否都存在
    if seed.__len__() != 3:
        return False, "分量缺失", None # err: 分量缺失

    # 三份量的计算结果
    result = {}

    # 这仨分量的长度，起始时间还不一样的，没法concatenate起来一步算完
    for trace in seed:
        wave_data = np.abs(trace.data)
        sr = trace.stats['sampling_rate']
        idx_p = int((p_time-trace.stats['starttime'])*sr)
        idx_s = int((s_time-trace.stats['starttime'])*sr)
        
        # 判断ps到时时间是否超出某分量的边界
        if idx_p <= 0 or idx_p >= wave_data.__len__() or idx_s <= 0 or idx_s >= wave_data.__len__():
            return False, "ps到达时间越界", None # err: ps到达时间越界
        
        # # 判断采样是否会导致超出边界
        # if idx_p+sec_to_decay*sr+sample*sr >= wave_data.__len__():
        #     return False, "时间越界", None # err: 时间越界

        # 判断ps到时间隔是否小于给定的衰减时常
        if idx_s-idx_p <= (sec_to_decay+sample)*sr:
            return False, "ps到时间隔过小", None # err: ps到时间隔过小

        # 保留P-S之间的数据
        wave_data = wave_data[idx_p:idx_s]


        # 计算相关性
        # https://thinkdsp-cn.readthedocs.io/zh_CN/latest/05-autocorrelation.html (5.2)

        y1 = wave_data[int(sr*sec_to_decay):int(sr*sec_to_decay)+int(sr*sample)]
        y2 = wave_data[:int(sr*sample)]
        corr = np.corrcoef(y1, y2, ddof=0)[0, 1]

        if trace.stats['channel'] == 'BHE':
            result['BHE'] = corr
        if trace.stats['channel'] == 'BHN':
            result['BHN'] = corr
        if trace.stats['channel'] == 'BHZ':
            result['BHZ'] = corr

    # 返回结果
    return 1, "", result

