#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/30
"""
"""
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

if __name__ == "__main__":
    print(__file__)
