import torch

# 第1步：准备输入
x = [
    [1, 0, 1, 0],               # Input 1
    [0, 2, 0, 2],               # Input 2
    [1, 1, 1, 1]                # Input 3
]
x = torch.tensor(x, dtype=torch.float)
print(x)


# 第2步：初始化参数
# Note: *通常在神经网络的初始化过程中，这些参数都是比较小的，一般会在Gaussian, Xavier and Kaiming distributions随机采样完成。*
w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]
w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
]
w_value = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float)
w_query = torch.tensor(w_query, dtype=torch.float)
w_value = torch.tensor(w_value, dtype=torch.float)

print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)



# 第3步：获取key, query和value
# Notes: *在我们实际的应用中，有可能会在点乘后，加上一个bias的向量。*
keys = x @ w_key
queries = x @ w_query
values = x @ w_value

print("Keys: \n", keys)
print("Queries: \n", queries)
print("Values: \n", values)


# 第4步：计算attention scores
attn_scores = queries @ keys.T
print("Attention Scores: \n", attn_scores)


# 第5步：计算softmax
from torch.nn.functional import softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)
print("Attention Scores Softmax: \n", attn_scores_softmax)

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)
print("Attention Scores Softmax: \n", attn_scores_softmax)


# 第6步：给value乘上score
weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
print("Weighted Values: \n", weighted_values)


# 第7步：给value加权求和获取output


# 第8步：重复步骤4-7, 获取output2, output3
outputs = weighted_values.sum(dim=0)
print("Outputs: \n", outputs)
