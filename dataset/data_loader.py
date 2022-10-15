import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from collections import defaultdict

Separator='==============================='




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

    list4=[]
    for i in range(len(list3)):
        random.shuffle(list3[i])
        list4.extend(list3[i])
    indices = torch.tensor(list4, dtype=torch.long)

    return indices


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.x_list, self.y_data = self.load_data_wrapper()


    def __len__(self):
        return len(self.y_data)


    def __getitem__(self, index):

        X = self.x_list[index]
        Y = self.y_data[index]


        return X, Y

    def load_data_wrapper(self):
        with open(self.root_dir, 'r+') as f1, open(self.label_dir, 'r+') as f2:
            x_data=[]
            y_data=[]
            x_list=[]
            f1_r = f1.readlines()
            f2_r = f2.readlines()
            id_list=[]
            id=0
            for x_line in f1_r:
                x_line = x_line.strip('\n')
                if x_line != Separator:
                    x_data.append(''.join(x_line))
                elif x_line == Separator and x_data != []:
                    x_list.append(x_data)
                    x_data = []
                    id+=1
                elif x_line == Separator and x_data == []:
                    id_list.append(id)
                    id+=1
            for j, y_line in enumerate(f2_r):
                if j+1 not in id_list:
                    y_line = y_line.strip('\n')
                    y_data.append(''.join(y_line))
          



            f1.close()
            f2.close()

        return x_list, y_data





class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        nums=[]
        for i in range(len(self.data)):
            num=len(self.data[i][0])
            nums.append(num)
        indices = group_shuffle(nums)
        return iter(indices)

    def __len__(self):
        return len(self.data)



class CustomBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                    i < len(sampler_list) - 1
                    and len(self.sampler.data[idx][0])
                    != len(self.sampler.data[sampler_list[i + 1]][0])
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                #else:
                    #batch = []
            i += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch



    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def is_None(batch):
    new_batch=[]
    for b in batch:
        feature, label=b
        if (feature is not None) and (label is not None):
            new_batch.append((feature,label))
    return new_batch



def collate_fn(batch):


    enc=OneHotEncoder()
    fea_batch=[]
    label_batch=[]

    for i in range(len(batch)):
        dic=batch[i]
        fea_batch.append(dic[0])
        label_batch.append(dic[1])

    # max_len is obtained from data statistics

    max_len=max()



    dim1 = []
    dim2 = []
    for i in range(len(fea_batch)):
        for j in range(len(fea_batch[0])):
            fea_arr = np.array(list(fea_batch[i][j])).reshape(-1, 1)
            fea_onehot = enc.fit_transform(fea_arr).toarray()
            fea_onehot = torch.tensor(fea_onehot, dtype=torch.float32)
            pad_arr = torch.zeros((max_len - len(fea_onehot), 4))
            fea_onehot_pad = torch.cat((fea_onehot, pad_arr), 0)
            dim1.append(fea_onehot_pad)
        dim11 = torch.stack(dim1)
        dim2.append(dim11)
        dim1 = []


    feature = torch.stack(dim2)



    label_list = []
    for i in range(len(label_batch)):
        label_arr = np.array(list(label_batch[i])).reshape(-1, 1)
        label_l_enc = enc.fit_transform(label_arr).toarray()
        label_l_enc = torch.tensor(label_l_enc)
        label_list.append(label_l_enc)

    label = torch.stack(label_list).argmax(dim=2)


    return feature, label









