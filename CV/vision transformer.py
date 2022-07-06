import torch
import torch.nn as nn
import torch.nn.functional as F

'''
将transformer应用到CV领域有难点：由于transformer的计算量与序列的平方是正比关系，若将一个像素点看作一个token，则计算量太大，而且一个像素点的信息太少。
故将一张图片分成几个patch，每个patch的大小相等且不变，每一个patch当作一个token作为transformer encoder的输入，这就是vision transformer的想法，用来做图像识别任务。
vision transformer的步骤：
    1 image --> patch -->embedding 将图片分成几个patch，patch展开，经过linear embedding，得到patch embedding(token embedding)
    2 类似bert，加入class token embedding 可以把该token当作query，用来提取图像patch的重要性，最后用来做分类
    3 position embedding
    4 将embedding输入transformer encoder
    5 classification head 将class token的输出提取出来，经过linear层，得到各分类的概率
'''

# 1 img --> embedding（两种方式）
def img2emb_patch(image, patch_size, weight):
    '''
    :param image: [batch_size, channels, h, w]
    :param patch_size: 每一个patch的大小
    :param weight: [patch_dim, model_dim], [patch_size*patch_size*channels, model_dim]
    :return: patch_embedding: [batchsize, seq_len(num of patchs), model_dim]
    '''
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2) # [1, 48, 4] --> [1, 4, 48] unfold函数就是把卷积的范围提取出来
    patch_embedding = patch @ weight
    return patch_embedding

def img2emd_conv(image, kernel, stride):
    '''
    :param image: [batch_size, channels, h, w]
    :param kernel: [output_channels, input_channels, kernel_h, kernel_w]
    :param stride: patch_size
    :return: patch_embedding: [batchsize, seq_len(num of patchs), model_dim]
    '''
    conv_output = F.conv2d(image, kernel, stride=stride) # conv_output.shape --> [batch_size, model_dim, h, w]
    batch_size, model_dim, h, w = conv_output.shape
    patch_embedding = conv_output.reshape((batch_size, model_dim, h*w)).transpose(-1, -2) # [batch_size, model_dim, seq_len(num of patchs)] --> [batch_size, seq_len(num of patchs), model_dim]
    return patch_embedding

if __name__ == '__main__':
    torch.manual_seed(0)

    # img2emb_patch函数的测试
    batch_size, channels, h, w = 1, 3, 8, 8
    patch_size = 4
    model_dim = 16
    weight = torch.randn(channels*patch_size*patch_size, model_dim)
    image = torch.randn(batch_size, channels, h, w)
    patch_embedding_patch = img2emb_patch(image, patch_size, weight)
    # print(patch_embedding_patch.shape)
    # print(patch_embedding_patch)

    # img2emd_conv的测试
    kernel = weight.transpose(0, 1).reshape(model_dim, channels, patch_size, patch_size) # [output_channels, input_channels, kernel_h, kernel_w]
    patch_embedding_conv = img2emd_conv(image, kernel, stride=patch_size)
    # print(patch_embedding_conv.shape)
    # print(patch_embedding_conv)

    # 2 构建class token embedding
    class_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
    token_embedding = torch.cat([class_token_embedding, patch_embedding_conv], dim=1)

    # 3 position embedding
    max_num_token = 10
    position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
    # print(position_embedding_table)
    seq_len = token_embedding.shape[1]
    # tile 和 unsqueeze的结果一样
    position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
    position_embedding_unsqu = torch.unsqueeze(position_embedding_table[:seq_len], 0)
    token_embedding_tile = token_embedding + position_embedding
    token_embedding_unsqu = token_embedding + position_embedding_unsqu
    # print(token_embedding_tile)
    # print(token_embedding_unsqu)

    # 4 经过transformer encoder --> 记得去看pytorch的transformer的API
    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    encoder_output = transformer_encoder(token_embedding_tile)
    # print(encoder_output.shape)

    # 5 分类 (Linear)
    num_classes = 7
    label = torch.randint(num_classes, (batch_size,))

    class_token_output = encoder_output[:, 0, :]
    linear_layer = nn.Linear(model_dim, num_classes)
    logits = linear_layer(class_token_output)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, label)
    print(loss)

