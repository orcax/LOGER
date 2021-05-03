def read_triplet(file_path):
    triplet2prob = {}
    with open(file_path) as fin:
        for line in fin:
            triplet = line.strip().split('\t')
            h, t, r, prob = triplet
            triplet2prob[(int(h), int(t))] = float(prob)
    return triplet2prob

import numpy as np
import torch
from metrics4rec import evaluate_all
import gzip

def hidden_gen(file, mln, transe, weight=1):
    c = 0
    candidates = np.load(file)['candidates']
    probs = np.zeros_like(candidates, dtype='float')
    for i in range(candidates.shape[0]):
        user = candidates[i, 0]
        for j in range(1, candidates.shape[1]):
            item = candidates[i, j]
            if (user, item) not in mln.keys():
                c += 1
            else:
                #  prob = mln[(user, item)]*weight + transe[(user, item)]
                prob = mln[(user, item)]
                probs[i, j] = prob
    return probs, c

def new_setting(file, mln, transe, weight=1):
    ui_scores, gt = {}, {}
    # for tri in mln.keys():
    #     h1, t1 = tri
    #     if tri not in transe.keys():
    #         continue
    #     if h1 not in ui_scores.keys():
    #         ui_scores[h1] = {}
    #     ui_scores[h1][t1] = mln[tri]

    mln_avg = sum(mln.values())/len(mln)
    for tri in transe.keys():
        h1, t1 = tri
        if h1 not in ui_scores.keys():
            ui_scores[h1] = {}
        if tri not in mln.keys():
            mln[tri] = 0
        ui_scores[h1][t1] = mln[tri]*weight + transe[tri]

    # print(len(ui_scores))
    with open(file) as fin:
        for line in fin:
            h, t, r = line.strip().split('\t')
            assert r == '0'
            if int(h) not in gt.keys():
                gt[int(h)] = []
            gt[int(h)].append(int(t))
    evaluate_all(ui_scores, gt, 10)
    cand = np.load('/common/users/yz956/kg/code/OpenDialKG/hidden50_cpa.npy')
    # cand = np.load('cand_pl.npy')
    s2 = {}
    for i in range(cand.shape[0]):
        s2[i] = {}
        for j in range(cand.shape[1]):
            s2[i][cand[i, j]] = 300 - j
    # print(s2[0])
    evaluate_all(s2, gt, 10)

    # c = np.load('/common/users/yz956/kg/code/KBRD/data/cpa/cpa/rec_test_candidate100.npz')['candidates']
    # m = 0
    # for i in range(c.shape[0]):
    #     if c[i][1] in gt[c[i][0]]:
    #         m += 1
    # print(m/c.shape[0])

            
mln = read_triplet('/common/users/yz956/kg/code/pLogicNet/record/cpa30/0/mln_10.31.txt')
transe = read_triplet('/common/users/yz956/kg/code/OpenDialKG/transe_cand_1000.txt')
transe = read_triplet('/common/users/yz956/kg/code/pLogicNet/record/cpa30/1/annotation_1000_htr.txt')
new_setting('/common/users/yz956/kg/code/KBRD/data/cpa/cpa/kg_test_triples_Cell_Phones_and_Accessories.txt', mln, transe, 0.3)
