from obspy import read
import numpy as np


def P_wave_complexity(miniseed_path, p_time, s_time, sec_to_decay=5):
    '''
        波形复杂度计算，参考《基于Bagging集成学习算法的地震事件性质识别分类》3.1章节

        args:
            miniseed_path: seed文件路径
            p_time: P波到达绝对时间 (obspy.core.utcdatetime.UTCDateTime)
            s_time: ...
            sec_to_decay: 经验系数：非天然震动事件Ｐ波能量衰减主要在5s左右完成。 default=5

        return: int, list
            错误码(int): 1无错误 -1分量缺失 -2时间越界 -3ps到时间隔过小
            计算结果(list)：对应在BHE BHN BHZ上的计算结果
    '''
    seed = read(miniseed_path)
    
    # 检查三个分量的波形数据是否都存在
    if seed.__len__() != 3:
        return -1, None # err: 分量缺失

    # 三份量的计算结果
    result = [0,0,0]

    # 这仨分量的长度，起始时间还不一样的，没法concatenate起来一步算完
    for trace in seed:
        wave_data = np.abs(trace.data)
        sr = trace.stats['sampling_rate']
        idx_p = int((p_time-trace.stats['starttime'])*sr)
        idx_s = int((s_time-trace.stats['starttime'])*sr)
        
        # 判断ps到时时间是否超出某分量的边界
        if idx_p <= 0 or idx_p >= wave_data.__len__() or idx_s <= 0 or idx_s >= wave_data.__len__():
            return -2, None # err: 时间越界

        # 判断ps到时间隔是否小于给定的衰减时常
        if idx_s-idx_p <= sec_to_decay*sr:
            return -3, None # err: ps到时间隔过小

        # 保留P-S之间的数据
        wave_data = wave_data[idx_p:idx_s]

        # 计算复杂度，积分按平均值计算
        C = np.average(wave_data[:int(sec_to_decay*sr)]**2)/np.average(wave_data[int(sec_to_decay*sr):]**2)

        if trace.stats['channel'] == 'BHE':
            result[0] = C
        if trace.stats['channel'] == 'BHN':
            result[1] = C
        if trace.stats['channel'] == 'BHZ':
            result[2] = C

    # 返回结果
    return 1, result

