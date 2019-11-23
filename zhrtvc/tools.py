import numpy as np
import collections as clt
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


def get_refs(data: list, num=3):
    """获取参考音频，和所有同一个说话人的embed余弦相似度最小。"""
    sim_mat = pairwise_distances(data, metric='cosine')
    sim_vec = np.mean(sim_mat, axis=0)
    ids = range(len(sim_vec))
    outs = [k for k, v in clt.Counter(dict(zip(ids, 1 - sim_vec))).most_common(num)]
    # outs = sorted(ids, key=lambda x: sim_vec[x], reverse=False)[:num]
    return outs


def get_speakers(data: list, num=3):
    """获取差异大的说话人，说话人之间相互的embed的余弦相似度最大。"""
    sim_mat = pairwise_distances(data, metric='cosine')
    sim_vec = np.mean(sim_mat, axis=0)
    targets = list(range(len(sim_vec)))
    idx = np.argmax(sim_vec)
    outs = [idx]
    targets.remove(idx)
    while 1:
        sim_vec = np.mean(sim_mat[outs], axis=0)
        idx = targets[int(np.argmax(sim_vec[targets]))]
        outs.append(idx)
        targets.remove(idx)
        if len(outs) >= num:
            break
    return outs


def judge_silence(data: np.array, threshold=0.1):
    """判断是否安静"""
    score = np.sqrt(np.var(data))
    return score <= threshold


def find_endpoint(data: np.array, threshold=0.1, min_silence_sec=1., hop_silence_sec=0.2):
    """找频谱的结束点，从前往后扫描，安静时长超过min_silence_sec则认为是结束点。"""
    dmax, dmin = np.max(data), np.min(data)
    vec = (data - dmin) / (dmax - dmin)  # data.sum(axis=1)
    window_length = int(min_silence_sec * 1000 / 12.5)  # hparams.frame_shift_ms
    hop_length = int(hop_silence_sec * 1000 / 12.5)
    for x in range(hop_length, len(vec) - window_length, hop_length):
        if judge_silence(vec[x:x + window_length], threshold):
            return x + hop_length
    return len(vec)


def find_silences(data: np.array, threshold=0.1, min_silence_sec=1., hop_silence_sec=0.2):
    """找频谱的安静区域，返回安静区域的起止位置列表。"""
    dmax, dmin = np.max(data), np.min(data)
    vec = (data - dmin) / (dmax - dmin)  # data.sum(axis=1)
    window_length = int(min_silence_sec * 1000 / 12.5)  # hparams.frame_shift_ms
    hop_length = int(hop_silence_sec * 1000 / 12.5)
    pairs = []
    start_point, end_point = 0, 0
    start_flag, end_flag = 1, 1
    for x in range(0, len(vec) - window_length, hop_length):
        if judge_silence(vec[x:x + window_length], threshold):
            if start_flag:
                start_point = x
                start_flag = 0
            end_point = x + window_length
        else:
            if not start_flag:
                pairs.append((start_point, end_point))
                start_flag = 1
    if not start_flag:
        pairs.append((start_point, end_point))
    return pairs


def find_start_end_points(data: np.array, threshold=0.1, min_silence_sec=1., hop_silence_sec=0.2):
    sils = find_silences(data, threshold=threshold, min_silence_sec=min_silence_sec, hop_silence_sec=hop_silence_sec)
    sidx, eidx = 0, len(data)
    if len(sils) >= 1:
        if sils[0][0] == 0:
            sidx = sils[0][1]
        if sils[-1][1] == len(data):
            eidx = sils[-1][0]
    return sidx, eidx
