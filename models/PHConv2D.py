import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PHConv2d(nn.Module):
    def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, cuda=False):
        super(PHConv2d, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, self.out_features // n, self.in_features // n, kernel_size, kernel_size))))
        self.kernel_size = kernel_size

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.F)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, A, F):
        siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
        siz2 = torch.Size(torch.tensor(F.shape[-2:]))
        res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
        siz0 = res.shape[:1]
        out = res.reshape(siz0 + siz1 + siz2)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size), device=self.A.device)
        for i in range(self.n):
            kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size,
                                                              self.kernel_size)
            H = H + kron_prod
        return H

    def forward(self, input):
        weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
        return F.conv2d(input, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=sqrt(5))
        nn.init.kaiming_uniform_(self.F, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.F)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)


