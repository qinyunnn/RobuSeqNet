import os
import sys
sys.path.append('../')

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.autograd import Variable
import argparse
from dataset import MyDataset, CustomSampler, CustomBatchSampler, collater
from Model import Model
from collections import Counter
from utils import statistics


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_data',
                        required=True,
                        help='Train data file',
                        default='MRR/examples/data/reads.txt')
    parser.add_argument('--ground_truth',
                        required=True,
                        help='Label file',
                        default='MRR/examples/data/reference.txt')
    parser.add_argument('--padding_length',
                        required=True,
                        type=int,
                        help='Maximum length of training data')
    parser.add_argument('--label_length',
                        required=True,
                        type=int,
                        help='Label length')
    parser.add_argument('--model_dir',
                        required=True,
                        help='Save model dir')
    parser.add_argument('--dim',
                        default=256,
                        type=int,
                        help='Feature dim')
    parser.add_argument('--lstm_hidden_dim',
                        default=256,
                        type=int,
                        help='Lstm hidden dim')
    parser.add_argument('--conv_dropout_p',
                        default=0.1,
                        help='Dropout of convolutional block')
    parser.add_argument('--rnn_dropout_p',
                        default=0.1,
                        help='Dropout of RNN block')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    gpu_id = 0
    gpu_str = 'cuda:{}'.format(gpu_id)
    device = torch.device(gpu_str if torch.cuda.is_available() else 'cpu')
    print(device)

    collate_fn = collater(args.padding_length)
    train_set=MyDataset(root_dir=args.train_data,
                        label_dir=args.ground_truth)

    train_cs=CustomSampler(data=train_set)
    train_bs=CustomBatchSampler(sampler=train_cs, batch_size=64, drop_last=False)
    train_dl=DataLoader(dataset=train_set, batch_sampler=train_bs, collate_fn=collate_fn)
    print('Finish loading')


    model=Model(noise_length=args.padding_length,
                    label_length=args.label_length,
                    dim=args.dim,
                    lstm_hidden_dim=args.lstm_hidden_dim,
                    conv_dropout_p=args.conv_dropout_p,
                    rnn_dropout_p=args.rnn_dropout_p).to(device)



    criterion = CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1)


    def train_loop(epoch, trainloader, model, criterion, optimizer):
        sample = {}
        for t in range(epoch):
            loss0 = []
            size = len(trainloader.dataset)
            for i, data in enumerate(trainloader):
                inputs, labels = data

                inputs, labels = Variable(inputs.float()).to(device), Variable(labels).to(device)

                outputs = model(inputs).type(torch.float32)
                loss = criterion(outputs, labels)
                y = outputs.argmax(dim=1)

                z = statistics(y, labels)
                optimizer.zero_grad()
                loss.requires_grad_()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    loss_fn, current = loss.item() / len(outputs), i * len(outputs)
                    print(f'Epoch {t + 1} \n ----------------------\nloss: {loss_fn:>7f}  [{current:>5d}/{size:>5d}]')
                    print(z)
                    loss0.append(loss_fn)
                    sample.setdefault(t + 1, []).append(loss0)
                    loss0 = []
            scheduler.step()
            torch.save(model.state_dict(), os.path.join(args.model_dir,'train_para_{}.pth'.format(epoch)))
        return sample


    epoch = 10
    train_loss = train_loop(epoch, train_dl, model, criterion, optimizer)
    torch.save(train_loss, 'train_loss_{}'.format(name))
    print('Finished Training')



if __name__ == '__main__':
    main()


