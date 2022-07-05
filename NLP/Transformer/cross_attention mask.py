import torch
import torch.nn.functional as F

'''
Q为decoder multi-head attention的输出
K为encoder的输出
cross_attention.shape --> [batchsize, tgt_seq_len, src_seq_len]
'''

torch.manual_seed(0)
batch_size = 3
max_src_seq_len = 5
max_tgt_seq_len = 5
src_len = torch.randint(2, max_src_seq_len, (batch_size,))
tgt_len = torch.randint(2, max_tgt_seq_len, (batch_size,))
# print(src_len)
# print(tgt_len)
'''
[tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1., 1., 1., 1.])] -->
[tensor([1., 1., 1., 1.]), tensor([1., 1., 0., 0.]), tensor([1., 1., 1., 1.])] -->
[tensor([[1., 1., 1., 1.]]), tensor([[1., 1., 0., 0.]]), tensor([[1., 1., 1., 1.]])] -->
tensor([[1., 1., 1., 1.],
        [1., 1., 0., 0.],
        [1., 1., 1., 1.]]) -->
        [3, 4, 1]
'''
vaild_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len)-L)), 0) for L in src_len], 0), -1)
vaild_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len)-L)), 0) for L in tgt_len], 0), -1)
# print(vaild_decoder_pos)
# print(vaild_decoder_pos.shape)

vaild_cross_pos_matrix = torch.bmm(vaild_decoder_pos, vaild_encoder_pos.transpose(-1,-2))
# print(vaild_cross_pos_matrix)
# print(vaild_cross_pos_matrix.shape)

invaild_cross_pos_matrix = 1 - vaild_cross_pos_matrix
mask_cross_attention = invaild_cross_pos_matrix.to(torch.bool)
# print(mask_cross_attention)