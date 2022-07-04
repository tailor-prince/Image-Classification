import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
batch_size = 3
model_dim = 8 # model_dim是input的维度，也是position的维度，两者后面会相加

# 序列最大长度，src_seq_len(source):输入的句子长度，tgt_seq_len（target):label的句子长度。（例如：source: deep learning; target:深度学习）
max_src_seq_len = 5
max_tgt_seq_len = 5

# 随机生成句子的长度
src_len = torch.randint(2, max_src_seq_len, (batch_size,))
tgt_len = torch.randint(2, max_tgt_seq_len, (batch_size,))

# 查看句子的长度 tensor([4, 2, 4]) tensor([2, 3, 2])
# print(src_len,'\n',tgt_len)

# 构建positional embedding，类似于word embedding，需要构建一个tabel，tabel的行代表位置（position），列代表维度，每一行就是一个位置的embedding。例如：第一行就是第1个位置的positional embeddeing
# pos_mat为tabel的行，pos_mat.shape --> [5, 1]，实际上tabel的维度为[5, 8]，但是由于广播机制，[5, 1]后面会变为[5, 8]
pos_mat = torch.arange(max(max_src_seq_len,max_tgt_seq_len)).reshape((max(max_src_seq_len,max_tgt_seq_len), -1))
# i_mat为tabel的列，这里只有4列，后面会分别sin,cos操作，变为8(model_dim)列。 [0, 2, 4, 6] --> [[0, 2, 4, 6]]
i_mat = torch.pow(1000, torch.arange(0, 8, 2).reshape((1,-1))/model_dim)

position_tabel = torch.zeros(max(max_src_seq_len,max_tgt_seq_len), model_dim) # 最初的tabel
position_tabel[:, 0::2] = torch.sin(pos_mat / i_mat)
position_tabel[:, 1::2] = torch.cos(pos_mat / i_mat)
# 查看position_tabel
# print(position_tabel)

# 根据position_tabel和position index，利用nn.Embedding得到position embedding
position_embedding = nn.Embedding(max(max_src_seq_len,max_tgt_seq_len), model_dim)
position_embedding.weight = nn.Parameter(position_tabel, requires_grad = False)

# method 2:另一种创建embedding的方法
position_embedding2 = nn.Embedding.from_pretrained(position_tabel)
print(position_embedding2.weight)

# 为每个句子创建位置索引，一个batch中句子长度应该一样长，长度为最大的那个句子长度
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32) # [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])] --> [tensor([[0, 1, 2, 3]]), tensor([[0, 1, 2, 3]]), tensor([[0, 1, 2, 3]])] --> [3, 4]
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)
# print(src_pos)

src_position_embedding = position_embedding(src_pos)
tgt_position_embedding = position_embedding(tgt_pos)
print(position_embedding.weight)
print(src_position_embedding)
print(tgt_position_embedding)


