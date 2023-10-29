import torch
import random

def statistics(x,y):
    z=x.eq(y).int()
    acc=torch.sum(z, dim=1)
    num=torch.sum(acc)
    dim=x.size(0)

    return num/dim

def is_None(batch):
    new_batch=[]
    for b in batch:
        feature, label=b
        if (feature is not None) and (label is not None):
            new_batch.append((feature,label))
    return new_batch


def getmaxlen(root_dir):
    max_len=0
    with open(root_dir, 'r+') as f:
        readline=f.readlines()
        for i, line in enumerate(readline):
            line=line.strip('\n')
            max_len=max(len(line), max_len)

    return max_len


def group_shuffle(data):
    tensor_data = torch.tensor(data, dtype=torch.long)
    ten, dices = torch.sort(tensor_data, descending=False)
    list1=list(ten)
    list2=list(dices)
    list3 = [list2[:1]]
    [list3[-1].append(t) if x == y else list3.append([t]) for x, y, s, t in zip(list1[:-1], list1[1:], list2[:-1], list2[1:])]
    #random.shuffle(list3)
    list4=[]
    for i in range(len(list3)):
        random.shuffle(list3[i])
        list4.extend(list3[i])
    indices = torch.tensor(list4, dtype=torch.long)

    return indices