import pickle

def read_path(file_path):
    triplet2path = {}
    cp, cn, c = 0, 0, 0
    with open(file_path) as fin:
        for line in fin:
            #  if c >= 110000000:
                #  c += 1
            #      break
            line = line.strip().split('\t')
            h, r0, mid0, r1, mid1, r2, t, rule_id = line
            key = (int(h), int(t))
            if t in itemp[int(h)]:
                cp += 1
            elif t in itemn[int(h)]:
                cn += 1
            c += 1
            if key not in triplet2path.keys():
                    triplet2path[key] = []
            triplet2path[key].append(line)
    print(cp, cn, c)
    return triplet2path

def read_item(train_path, hidden_path):
    train, hidden = [], []
    with open(train_path) as ft:
        for line in ft:
            line = line.strip().split('\t')
            train.append(line[1:])
    with open(hidden_path) as fh:
        for line in fh:
            line = line.strip().split('\t')
            hidden.append(line)

    assert len(train) == len(hidden)
    return train, hidden

def write_triplets(mln, out_file):
    with open(out_file, 'w') as fo:
        for triplet in mln.keys():
            h, t = triplet
            line = str(h) + '\t' + str(t) + '\n'
            fo.write(line)

def conpath(mln, out_file):
    tp3 = []
    for triplet in mln.keys():
        paths = mln[triplet]
        tmp = []
        for path in paths:
            #  print(path)
            h, r0, mid0, r1, mid1, r2, t = path
            tmp.append([int(r0), int(r1), int(r2)])
        tp3.append(tmp)
    pickle.dump(tp3, open(out_file, 'wb'))
            

def write_path(mln, itemp, itemn, out_file):
    for triplet in mln.keys():
        p_line = '(['
        paths = mln[triplet]
        for path in paths:
            #  print(path)
            h, r0, mid0, r1, mid1, r2, t, rule_id = path
            n0, n1 = int(mid0), int(mid1)
            if n0 < 61254:
                t0 = '0'
            elif n0 < 108858:
                t0 = '1'
            else:
                t0 = '2' 
            if n1 < 61254:
                t1 = '0'
            elif n1 < 108858:
                t1 = '1'
            else:
                t1 = '2'
            if t in itemp[int(h)]:
                k = '1'
            elif t in itemn[int(h)]:
                k = '1'
            else:
                k = '1'
                # print(h, t, '-1')
            p_temp = '([[' + h + ', 0, ' + r0 + '], [' + mid0 + ', ' + t0 + ', ' + r0 + '], [' + mid1 + ', ' + t1 + ', ' + r1 + '], [' + t + ', 1, ' + r2 + ']], 4)'
            p_line = p_line + p_temp + ', '
        p_line = p_line[:-2]
        p_line += '], ' + k + ')\n'
        # print(p_line)
        with open(out_file, 'a') as fo:
            fo.write(p_line)

def read_triplet(file_path, hidden):
    triplet2prob = {}
    with open(file_path) as fin:
        for line in fin:
            triplet = line.strip().split('\t')
            h, t, r, prob = triplet
            if t in hidden[int(h)]: 
                triplet2prob[(h, t)] = float(prob)
    return triplet2prob

def tri2idx(file_path):
    tri2idx = {}
    with open(file_path) as fin:
        i = 0
        for line in fin:
            h = line.strip().split('\t')[0]
            t = line.strip().split('\t')[1]
            tri2idx[(h, t)] = i
            i += 1
    return tri2idx

def change_target(file_path, out_path, tri2idx):
    path = []
    c = 0
    with open(file_path) as fin:
        for line in fin:
            path.append(line.strip())
    for key in triplet2prob.keys():
        if triplet2prob[key] > 0.2:
            if key not in tri2idx.keys():
                c += 1
                continue
            with open('/common/users/yz956/kg/code/KBRD/data/cpa/cpa/complete_1.txt', 'a') as fout:
                h, t = key
                fout.write(str(h) + '\t' + str(t) + '\n')
    #         i = tri2idx[key]
    #         assert int(path[i][-2]) == 0
    #         path[i] = path[i][:-2] + '1)'
    # print(c)
    # with open(out_path, 'w') as fo:
    #     for p in path:
    #         fo.write(p + '\n')

def write_oneline(paths, out_file):
    p_line = '(['
    for path in paths:
        #  print(path)
        h, r0, mid0, r1, mid1, r2, t = path
        n0, n1 = int(mid0), int(mid1)
        if n0 < 61254:
            t0 = '0'
        elif n0 < 108858:
            t0 = '1'
        else:
            t0 = '2' 
        if n1 < 61254:
            t1 = '0'
        elif n1 < 108858:
            t1 = '1'
        else:
            t1 = '2'
        if t in itemp[int(h)]:
            k = '1'
        elif t in itemn[int(h)]:
            k = '0'
        else:
            k = -1
            print(h, t, '-1')
        p_temp = '([[' + h + ', 0, ' + r0 + '], [' + mid0 + ', ' + t0 + ', ' + r0 + '], [' + mid1 + ', ' + t1 + ', ' + r1 + '], [' + t + ', 1, ' + r2 + ']], 4)'
        p_line = p_line + p_temp + ', '
    p_line = p_line[:-2]
    p_line += '], ' + k + ')\n'
    # print(p_line)
    with open(out_file, 'a') as fo:
        fo.write(p_line)

def read_again(file_path, triplet_path, out_file):
    keys = []
    with open(triplet_path) as ft:
        for line in ft:
            h = line.strip().split('\t')[0]
            t = line.strip().split('\t')[1]
            keys.append((h, t))
    #  for key in keys:
        #  paths = []
        #  kh, kt = key
        #  with open(file_path) as fin:
            #  for line in fin:
                #  line = line.strip().split('\t')
                #  h, r0, mid0, r1, mid1, r2, t = line
                #  if kh == h and kt == t:
                    #  paths.append(line)
    #      write_oneline(paths, out_file)
    l = len(keys)
    m1, m2 = {}, {}
    for i in range(l):
        if i >= l/10:
            m1[keys[i]] = []
    with open(file_path) as fin:
        for line in fin:
            line = line.strip().split('\t')
            h, r0, mid9, r1, mid1, r2, t, rule_id = line
            if (h, t) in m1.keys():
                m1[(h, t)].append(line)
    return m1

def prob_test(tri2prob):
    c = 0
    for key in tri2prob.keys():
        if tri2prob[key] > 0.85:
            c += 1
    print(c/len(tri2prob))



itemp, itemn = read_item('/common/users/yz956/kg/code/KBRD/data/cpa/cpa/sample_pre.txt', '/common/users/yz956/kg/code/KBRD/data/cpa/cpa/hidden50_cand1com.txt')
mln = read_path('/common/users/yz956/kg/code/pLogicNet/record/cpa10/paths10_cpa_sample.txt')
# tri2prob = read_triplet('/common/users/yz956/kg/code/PathCon/src/scores_n.txt', itemn)
# prob_test(tri2prob)
# mln = read_again('/common/users/yz956/kg/code/pLogicNet/record/paths50_transe_9.2.txt', '/common/users/yz956/kg/code/pLogicNet/record/train_triplet_9.2.txt', '/common/users/yz956/kg/code/pLogicNet/record/train_path_9.2_1.txt')
write_path(mln, itemp, itemn, '/common/users/yz956/kg/code/pLogicNet/record/cpa10/train_path10_sample.txt')
#write_triplets(mln, '/common/users/yz956/kg/code/pLogicNet/record/test_triplet_1000cand_1.txt')
#  triplet2prob = read_triplet('/common/users/yz956/kg/code/pLogicNet/record/2020-08-10_09:04:33.089757/0/pred_mln_test_0.1.txt', itemn)
#  tri2idx = tri2idx('/common/users/yz956/kg/code/pLogicNet/record/train_triplet.txt')
#  change_target('/common/users/yz956/kg/code/pLogicNet/record/train_path.txt', '/common/users/yz956/kg/code/pLogicNet/record/train_path_1.txt', tri2idx)

# conpath(mln, '/common/users/yz956/kg/code/PathCon/data/cpa/cache/train_paths_3.pkl')

# import random
# def read_sample(file_path):
#     triplet2path = {}
#     count = len(open(file_path).readlines())
#     r = [i for i in range(count)]
#     random.shuffle(r)
#     r = r[:count/10]
#     cp, cn, c = 0, 0, 0
#     with open(file_path) as fin:
#         for line in fin:
#             if c not in r:
#                 c += 1
#                 continue
#             line = line.strip().split('\t')
#             h, r0, mid0, r1, mid1, r2, t, rule_id = line
#             key = (int(h), int(t))
#             if t in itemp[int(h)]:
#                 cp += 1
#             elif t in itemn[int(h)]:
#                 cn += 1
#             c += 1
#             # if key not in triplet2path.keys():
#                     # triplet2path[key] = []
#             # triplet2path[key].append(line)
#     print(cp, cn, c)
#     return triplet2path

# mln = read_sample('/common/users/yz956/kg/code/pLogicNet/record/paths50_transe_9.5.txt')
