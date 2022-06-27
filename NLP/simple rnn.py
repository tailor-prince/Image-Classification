import torch
import torch.nn as nn

'''
单向单层RNN
参数维度：
    input_shape: [batch_size, sequence length, input_size]
    ho_shape: [batch size, D*num_layers, hidden_size] --> 初始的memeory的值，D表示是不是双向
    output_shape: [batch_size, sequence length, D*hidden_size] --> output包含了每一个单词的最后输出，D*hidden_size是因为当网络为双向时，output会concatten forward和backward的结果
    hn_shape: [D*num_layers, batch_size, hidden_size]
函数实现参数维度：
    weight_ih: [hidden_size, input_size]
    weight_hh: [hidden_size, hidden_size]
    bias_ih: [hidden_size]
    bias_hh: [hidden_size]
    
weight_ih * input + weight_hh * ho: [hidden_size]
'''

def rnn_forward(weight_ih, weight_hh, bias_ih, bias_hh, h0, input):
    batch_size, sequence_length, input_size = input
    h_out = torch.zeros(batch_size, sequence_length, weight_hh.shape[0]) # 初始化最后的output矩阵

    for t in range(sequence_length):
        x_t = input[:,t,:].unsqueeze(-1) # 获得某一个单词的输入 [batch_size, input_size] --> [batch_size, input_size, 1]
        weight_ih = weight_ih.unsqueeze(0).tile(batch_size, 1, 1) # [hidden_size, input_size] --> [batch_size, hidden_size, input_size] tile?
        weight_hh = weight_hh.unsqueeze(0).tile(batch_size, 1, 1) # [hidden_size, hidden_size] --> [batch_size, hidden_size, hidden_size]

        w_times_x = torch.bmm(weight_ih, x_t).squeeze(-1) # bmm是为了做不受batch_size影响的矩阵乘法, [batch_size, hidden_size, 1] --> [batch_size, hidden_size]
        w_times_h = torch.bmm(h0, weight_hh).squeeze(1) # [batch size, D*num_layers, hidden_size] --> [batch_size, hidden_size]

        h0 = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh) # [batch_size, hidden_size]
        h_out[:,t,:] = h0
        h0 = h0.unsqueeze(1)

    return h_out, h0

