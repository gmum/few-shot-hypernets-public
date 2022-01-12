import torch
import torch.nn as nn


class QueryConv(nn.Module):
    def __init__(self, hn_queries_convolution_output_dim):
        super(QueryConv, self).__init__()
        self.conv = nn.Conv1d(85, 85, kernel_size=3)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 2, kernel_size=(5, 5)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(2, 3, kernel_size=(3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 5, kernel_size=(2, 2)),
        #     nn.ReLU(inplace=True)
        # )
        # self.fc = nn.Linear(5 * 15 * 15, hn_kernel_convolution_output_dim)

    # def forward(self, x):
    #     x = self.conv(x)
    #     x = torch.flatten(x, start_dim=1, end_dim=-1)
    #     out = self.fc(x)
    #     return out

    def forward(self, x):
        return