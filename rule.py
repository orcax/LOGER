def read_rule(file):
    w = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            w.append(float(line[-1]))
    return w

w = read_rule('/common/users/yz956/kg/code/pLogicNet/record/9-1/3/rule_0.3.txt')
ws = sorted(w, reverse=True)
print(ws)
