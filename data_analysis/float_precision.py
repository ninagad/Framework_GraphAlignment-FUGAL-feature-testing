import sys
import torch

print(sys.float_info)
C = torch.ones(1,1,dtype=torch.float64)
a = torch.ones(1, dtype=torch.float64)
b = torch.ones(1, dtype=torch.float64)
u = torch.ones(1, dtype=torch.float64)

#
K = torch.empty(C.shape, dtype=torch.float64)
torch.div(C, - 0.00140888187586813, out=K)
torch.exp(K, out=K)

KTu = torch.matmul(u, K)
v = torch.div(b, KTu)
Kv = torch.matmul(K, v)
u = torch.div(a, Kv)

print(f'{torch.any(KTu == 0)=}')
print(f'{torch.any(torch.isnan(u))=}')
print(f'{torch.any(torch.isnan(v))=}')
print(f'{torch.any(torch.isinf(u))=}')
print(f'{torch.any(torch.isinf(v))=}')

print(f'{torch.is_nonzero(K)=}')
print(f'{torch.is_nonzero(KTu)=}')

print(f"{K=}")
print(f'{KTu=}')
print(f'{v=}')

print(f'{1./(2.2250738585072014e-308)=}')



