import json
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm

"""
@Author: Bo Fei
"""

# 数据库路径
dataset_path = r'D:\AkagawaTsurunaki\Dataset\.input\labeled_snsccb.json'
model_save_path = r'D:\AkagawaTsurunaki\Dataset\Classifier\models\feebor.parameters'
pretrained_name = r'D:\AkagawaTsurunaki\Models\bert-base-chinese'
output_dim = 28  # 因为我们划分了28种情感, 所以这里选择28
split = 0.9


def read_file(file_name):
    # 读取评论信息
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打乱数据集
    random.shuffle(data)

    # 创建新的字段 "combined_text"，将 "post" 和 "response" 连接在一起，没有空格分割
    for entry in data:
        entry["combined_text"] = entry["post"][:200] + entry["response"][:200]

    # 删除重复的评论内容
    data = [dict(t) for t in {tuple(d.items()) for d in data}]

    # 转换为DataFrame
    df = pd.DataFrame(data)

    return df  # 返回DataFrame


class BERTClassifier(nn.Module):

    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, output_dim, pretrained_name):
        super(BERTClassifier, self).__init__()

        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(pretrained_name)

        # 外接全连接层
        in_features = 768
        self.mlp = nn.Linear(in_features, output_dim)

    def forward(self, tokens_X):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)

        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析
        return self.mlp(res[1])


"""
评估函数，用以评估数据集在神经网络下的精确度
"""


def evaluate(net, comments_data, labels_data):
    sum_correct, i = 0, 0

    while i <= len(comments_data):
        comments = comments_data[i: min(i + 8, len(comments_data))]

        tokens_X = tokenizer(comments, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
            device=device)

        res = net(tokens_X)  # 获得到预测结果

        y = torch.tensor(labels_data[i: min(i + 8, len(comments_data))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()  # 累加预测正确的结果
        i += 8

    return sum_correct / len(comments_data)  # 返回(总正确结果/所有样本)，精确率


"""
训练bert_classifier分类器
"""


def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels,
                          device, epochs):
    global model_save_path
    max_acc = 0.5  # 初始化模型最大精度为0.5

    # 先测试未训练前的模型精确度
    train_acc = evaluate(net, train_comments, train_labels)
    test_acc = evaluate(net, test_comments, test_labels)

    # 输出精度
    print('--epoch', 0, '\t--train_acc:', train_acc, '\t--test_acc', test_acc)

    # 累计训练数据 epochs 次，优化模型
    for epoch in tqdm(range(epochs)):

        i, sum_loss = 0, 0  # 每次开始训练时， i 为 0 表示从第一条数据开始训练

        # 开始训练模型
        while i < len(train_comments):
            comments = train_comments[i: min(i + 8, len(train_comments))]  # 批量训练，每次训练8条样本数据

            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)

            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + 8, len(train_comments))]).reshape(-1).to(device=device)

            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度

            sum_loss += l.detach()  # 累加损失
            i += 8  # 样本下标累加

        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)

        # 输出精度
        print('\n--epoch', epoch + 1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)

        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc

            # 保存模型
            torch.save(net.state_dict(), model_save_path)


comments_data = read_file(dataset_path)
split_line = int(len(comments_data) * split)

# 划分训练集
train_comments = list(comments_data['combined_text'][:split_line])
train_labels = list(comments_data['label'][:split_line])
# 划分测试集
test_comments = list(comments_data['combined_text'][split_line:])
test_labels = list(comments_data['label'][split_line:])

device = d2l.try_gpu()  # 获取GPU

net = BERTClassifier(output_dim, pretrained_name)  # BERTClassifier分类器，因为最终结果为28分类，所以输出维度为28，代表概率分布
net = net.to(device)  # 将模型存放到GPU中，加速计算

# 定义tokenizer对象，用于将评论语句转化为BertModel的输入信息
tokenizer = BertTokenizer.from_pretrained(pretrained_name)

loss = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)  # 小批量随机梯度下降算法

train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, device,
                      20)
torch.save(net.state_dict(), model_save_path)  # 最终保存模型
