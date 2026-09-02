from codon.block.param import BoundedTensor
import torch

# 单个标量，区间 [0, 1]，初始值在区间中点 0.5
a = BoundedTensor.single(bound=(0.0, 1.0))
print(a.shape)          # torch.Size([])，零维标量
print(float(a.fresh())) # 0.5

# 一组 5 个值，区间 [-1, 1]，初始值都在中点 0
v = BoundedTensor.bounded(5, bound=(-1.0, 1.0))
print(v.shape)          # torch.Size([5])
print(v)

# 参与运算
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = v + x
print(y)

loss = (y * y).sum()
loss.backward()
print(v.raw.grad)       # 梯度流回 raw