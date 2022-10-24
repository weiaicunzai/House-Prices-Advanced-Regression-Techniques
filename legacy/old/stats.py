

import numpy as np

#a = 'iou: 0.767520, f1: 0.809464, recall: 0.814330, precision: 0.855400, dice: 0.838147, hausdorff: 79.516914'

def str2float(line):
    line = line.strip()
    line = line.split(', ')
    line = [float(l.split(': ')[1]) for l in line]
#    print(line)
    return line

res = []
with open('res3.txt') as f:
    for line in f.readlines():
        if line.strip().startswith('#'):
            continue

        if not line.strip():
            continue
        #line = line.strip().split('|')[1:-1]
        line = str2float(line)
        #line = line.strip().split()
        #line = [float(n) for n in line]
        print('line', line)
        res.append(line)


res = np.array(res)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
means = res.mean(axis=0)
stds = res.std(axis=0)
print(means)
print(stds)

res_str = '|'
for m, s in zip(means, stds):
    res_str += '{:0.4f}±{:0.3f}|'.format(m, s)

print(res_str)