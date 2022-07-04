import torch.nn.functional as F
import torch

'''
一个batch，句子的长度应该相同，使用padding，
可是在计算attention score时，句子中原本的单词不应该和padding的单词计算，
使用mask,使得这种位置的attenion score变为负无穷大，在经过softmax函数，该位置就会变为0

encoder_input.shape --> [batch_size, src_len, model_dim] src_len: max_batch_seq_len
attention_score.shape --> [batch_size, src_len, src_len] ==> mask.shape --> [batch_size, src_len, src_len] 其中mask中的值为1或-inf
'''
batch_size = 3
torch.manual_seed(0)
max_src_seq_len = 5
src_len = torch.randint(2, max_src_seq_len, (batch_size,))
# print(src_len)

# vaild_encoder_pos_matrix --> mask的初步矩阵，其中为1的部分表示两者有attention score，为0的部分表示无attention score。
# 需要注意的是mask矩阵元素为bool类型，True代表要mask（无attention score），False则无需变动（有attention score）。因此需要1-vaild_encoder_pos_matrix
vaild_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len-L))), 0) for L in src_len], 0), -1)
vaild_encoder_pos_matrix = torch.bmm(vaild_encoder_pos, vaild_encoder_pos.transpose(1,2))
# vaild_encoder_pos_matrix = torch.bmm(vaild_encoder_pos.transpose(1,2), vaild_encoder_pos)

# print(vaild_encoder_pos)
# print(vaild_encoder_pos.shape)
# print(vaild_encoder_pos_matrix)
# print(vaild_encoder_pos_matrix.shape)

# 真正的mask矩阵
invaild_encoder_pos_matrix = 1 - vaild_encoder_pos_matrix
# 把矩阵元素变成bool类型
mask_encoder_self_attention = invaild_encoder_pos_matrix.to(torch.bool)
#print(mask_encoder_self_attention)

# 随机初始化一个attention score，就是QK
score = torch.randn(batch_size, max(src_len), max(src_len))

# mask_score中mask的地方还是很大的负数，经过softmax后，这些位置变成0，即使有些行的概率值相等，也不用担心，后期不会对这种地方进行梯度更新。
mask_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(mask_score, -1)

print(score)
print(mask_score)
print(prob)