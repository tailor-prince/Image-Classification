import torch
import torch.nn.functional as F

'''
decoder每次的输入是一个下三角矩阵，mask就是下三角矩阵，可以先构建一个方针，再构建下三角矩阵

'''

torch.manual_seed(0)
batch_size = 3
max_tgt_seq_len = 5
tgt_len = torch.randint(2, max_tgt_seq_len, (batch_size,))
# print(tgt_len)

vaild_decoder_tri_matrix = torch.cat([torch.unsqueeze(torch.tril(F.pad(torch.ones((L, L)), (0, max(tgt_len)-L, 0, max(tgt_len)-L))), 0) for L in tgt_len])
# print(vaild_decoder_tri_matrix)
invaild_decoder_tri_matrix = 1 - vaild_decoder_tri_matrix
invaild_decoder_tri_matrix = invaild_decoder_tri_matrix.to(torch.bool)
print(invaild_decoder_tri_matrix)
