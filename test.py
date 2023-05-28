# import numpy as np

# # 定义输入数组和卷积核
# a = np.array([1, 1, 2, 3, 4, 5, 5])
# v = np.array([1, 2, 3])

# # 计算线性卷积
# res = np.convolve(a, v, mode='full')

# print(res)


















# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Before call")
#         result = func(*args, **kwargs)
#         print("After call")
#         return result
#     return wrapper

# @my_decorator
# def add(a, b):
#     print('in func')
#     return a + b

# print(add(1, 3))




import torch
# x=torch.ones(2,1)
# y=torch.ones(1,3)
# z=(x==y)
# z=z[...,None]
# z=z.squeeze(2)
# print(z.shape)
# print(z.data)
import random
random.seed(0)
torch.manual_seed(seed=0)
print(torch.initial_seed())

li = [1,2,3]
random.shuffle(li)
print(*li)
print(random.uniform(-320,960))