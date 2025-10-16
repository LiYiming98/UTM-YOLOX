import torch

a = torch.tensor([-2,-3,-4])
a =  0.09 *a
a = torch.sigmoid(a)
print(a)
