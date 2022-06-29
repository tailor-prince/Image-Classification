import torch
import torch.nn as nn

'''
i_t = sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
f_t = sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
o_t = sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
c_t = f_t .dot c_{t-1} + i_t .dot g_t
h_t = o_t .dot tanh(c_t)  (.dot means elements mul)
单向单层LSTM
实例化对象参数:
    input_size: 每个单词的维度
    hidden_size: LSTM cell的个数
    batch_first: the input and output tensors are provided as (batch, seq, feature),note that this does not apply to hidden or cell states.
    dropout: introduces a Dropout layer on the outputs of each LSTM layer except the last layer.  
    proj_size:
对象参数:
    input_shape: [batch_size, sequence length, input_size]
    h_0_shape: [D*num_layers, batch size, hidden_size]
    c_0_shape: [D*num_layers, batch size, hidden_size]
    output_shape: [batch_size, sequence length, D*hidden_size]
    h_n_shape: [D*num_layers, batch_size, hidden_size]
    c_n_shape: [D*num_layers, batch_size, hidden_size]
函数实现参数:
    w_ih: [4 * hidden_size, input_size]
    w_hh: [4 * hidden_size, hidden_size]
    b_ih: [4 * hidden_size]
    b_hh: [4 * hidden_size]
'''

def lstm_forward(w_ih, w_hh, b_ih, b_hh, input, init_states):
    # 得到一些初始条件
    batch_size, sequence_length, input_size = input.shape
    hidden_size = w_hh.shape[1]
    h0, c0 = init_states #初始的h0,c0 [batch size, hidden_size], [batch size, hidden_size]
    h_out = torch.zeros(batch_size, sequence_length, hidden_size) # 初始一个output

    w_ih = w_ih.unsqueeze(0).tile(batch_size, 1, 1) #[4 * hidden_size, input_size] --> [batch_size, 4 * hidden_size, input_size]
    w_hh = w_hh.unsqueeze(0).tile(batch_size, 1 ,1) #[4 * hidden_size, hidden_size] --> [batch_size, 4 * hidden_size, hidden_size]

    for t in range(sequence_length):
        x = input[:,t,:].unsqueeze(-1) # [batch_size, input_size] --> [batch_size, input_size, 1]
        h0 = h0.unsqueeze(-1) # [batch size, hidden_size] --> [batch size, hidden_size, 1]

        w_times_x = torch.bmm(w_ih, x).squeeze(-1) # [batch_size, 4 * hidden_size, 1] --> [batch_size, 4 * hidden_size]
        w_times_h = torch.bmm(w_hh, h0).squeeze(-1) # [batch_size, 4 * hidden_size, 1] --> [batch_size, 4 * hidden_size]

        # i_t:输入门, f_t:遗忘门, g_t:输入, o_t:输出门 [bach_size, hidden_size]
        i_t = torch.sigmoid(w_times_x[:, :hidden_size] + b_ih[:hidden_size] + w_times_h[:, :hidden_size] + b_hh[:hidden_size])
        f_t = torch.sigmoid(w_times_x[:, hidden_size:2*hidden_size] + b_ih[hidden_size:2*hidden_size] + w_times_h[:, hidden_size:2*hidden_size] + b_hh[hidden_size:2*hidden_size])
        g_t = torch.tanh(w_times_x[:, 2*hidden_size:3*hidden_size] + b_ih[2*hidden_size:3*hidden_size] + w_times_h[:, 2*hidden_size:3*hidden_size] + b_hh[2*hidden_size:3*hidden_size])
        o_t = torch.sigmoid(w_times_x[:, 3*hidden_size:] + b_ih[3*hidden_size:] + w_times_h[:, 3*hidden_size:] + b_hh[3*hidden_size:])

        c0 = f_t * c0 + i_t * g_t # [bach_size, hidden_size] 这里的c0不再是初始的c0了，而是代表这一时刻的c0，将参加下一时刻的计算
        h0 = o_t * torch.tanh(c0) # [bach_size, hidden_size] 这里的h0不再是初始的h0了，而是代表这一时刻的h0，将参加下一时刻的计算

        h_out[:,t,:] = h0

    return h_out, (h0.unsqueeze(0), c0.unsqueeze(0))

# test

# pytorch版本LSTM
model_lstm = nn.LSTM(5, 3, batch_first=True)
input = torch.randn(2, 4, 5)
output_lstm, (hn_lstm, cn_lstm) = model_lstm(input)
print(output_lstm, '\n', hn_lstm, '\n', cn_lstm)
print(output_lstm.shape, hn_lstm.shape, cn_lstm.shape)

# 自我实现的LSTM
# 查看模型参数名称
# for k, v in model_lstm.named_parameters():
#     print(k, v)
h0 = torch.zeros(2, 3)
c0 = torch.zeros(2, 3)
output_lstm_forward, (hn_lstm_forward, cn_lstm_forward) = lstm_forward(model_lstm.weight_ih_l0, model_lstm.weight_hh_l0, model_lstm.bias_ih_l0, model_lstm.bias_hh_l0, input, (h0, c0))
print('-' * 50)
print(output_lstm_forward, '\n', hn_lstm_forward, '\n', cn_lstm_forward)
print(output_lstm_forward.shape, hn_lstm_forward.shape, cn_lstm_forward.shape)

