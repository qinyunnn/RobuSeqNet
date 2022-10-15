import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple
from conformer import ConformerBlock


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)



class Conv(nn.Module):
    def __init__(self,
                 in_channels=4,
                 score_dropout=0.1,
                 ) -> None:
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.sequential1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels // 2, out_channels=in_channels // 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(score_dropout)
        )


    def forward(self, input: Tensor) -> Tensor:
        b = input.size(dim=0)
        n = input.size(dim=1)
        l = input.size(dim=2)
        d = input.size(dim=3)
        input = input.view(b*n, l, d)
        input = torch.transpose(input, 1, 2)
        output = self.sequential1(input)
        output = output.view(b, n, 1, l)
        return output.squeeze(dim=2)




class AttnScore(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self,
                 in_features: int,
                 out_features=64,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AttnScore, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.Vc = Parameter(torch.empty((out_features, 1), **factory_kwargs))
        self.Kc = Parameter(torch.empty(1, **factory_kwargs))
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound1 = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound1, bound1)
        init.kaiming_uniform_(self.Vc, a=math.sqrt(5))
        fan_in2, _ = init._calculate_fan_in_and_fan_out(self.Vc)
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        init.uniform_(self.Kc, -bound2, bound2)

    def forward(self, input: Tensor) -> Tensor:
        output = torch.einsum('i j k, k l -> i j l', input, self.weight)
        output = output+self.bias
        output = self.relu(output)
        output2 = torch.einsum('i j l, l t-> i j t', output, self.Vc)
        output2 = output2+self.Kc
        output3 = self.softmax(output2)
        return output3.squeeze(dim=2)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features)




class Weight(nn.Module):
    def __init__(self,
                 noise_length: int,
                 ) -> None:
        super(Weight, self).__init__()

        self.Conv = Conv(in_channels=4)
        self.score = AttnScore(in_features=noise_length)

    def forward(self, input: Tensor) -> Tensor:
        output=self.Conv(input)
        output=self.score(output)
        output = torch.einsum('i j,i j k l -> i j k l', output, input)
        output = torch.sum(output, dim=1)

        return output



class Linear(nn.Module):
    def __init__(self,
                 noise_length: int,
                 label_length: int
                 ) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(noise_length, label_length),
            nn.ReLU()
        )

    def forward(self, input: Tensor) -> Tensor:
        output = torch.transpose(input, 1, 2)
        output = self.linear(output)
        return output.transpose(1,2)






class Conv2dUpampling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_dropout_p: int,
                 kernel_size=3
                 ) -> None:
        super(Conv2dUpampling, self).__init__()
        padding=calc_same_padding(kernel_size)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(p=conv_dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.dropout(outputs)
        return outputs







class RNNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 lstm_hidden_dim: int,
                 rnn_dropout_p=0.,
                 ) -> None:
        super(RNNBlock, self).__init__()
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_dim, num_layers=2, bidirectional=False,
                           batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=4)
        self.dropout = nn.Dropout(rnn_dropout_p)


    def forward(self, input: Tensor) -> Tensor:
        output, _ = self.rnn(input)
        output = self.linear(output)
        output = self.dropout(output)

        return output



class Model(nn.Module):
    def __init__(self,
                 noise_length: int,
                 label_length: int,
                 dim: int,
                 lstm_hidden_dim: int,
                 res_dropout_p=0.,
                 conv_dropout_p=0.,
                 rnn_dropout_p=0.,
                 ) -> None:
        super(Model, self).__init__()
        self.addweight = Weight(noise_length=noise_length)
        self.linear = Linear(noise_length=noise_length, label_length=label_length)
        self.upsampling = Conv2dUpampling(in_channels=1, out_channels=dim//4, conv_dropout_p=conv_dropout_p)
        self.res2net = Res2Net(in_channels=dim,
                               res_dropout_p=res_dropout_p) # option: before_num, dim_expansion
        self.conformer = ConformerBlock(dim=dim,
                                        dim_head=64,
                                        heads=8,
                                        ff_mult=4,
                                        conv_expansion_factor=2,
                                        conv_kernel_size=31,
                                        attn_dropout=0.1,
                                        ff_dropout=0.1,
                                        conv_dropout=0.1
                                        )
        self.rnnblock = RNNBlock(in_channels=dim,
                                      lstm_hidden_dim=lstm_hidden_dim,
                                      rnn_dropout_p=rnn_dropout_p) # option: lstm_hidden_dim [64,128,256]

    def forward(self, input: Tensor) -> Tensor:
        output = self.addweight(input)
        output = self.linear(output)
        output = self.upsampling(output)
        output = self.conformer(output)
        output = self.rnnblock(output)
        output = output.permute(0, 2, 1)

        return output




