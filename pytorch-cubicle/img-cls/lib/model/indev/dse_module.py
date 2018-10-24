from torch import nn


class DSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x, xr):
        b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.avg_pool(x).view(b, c) + self.avg_pool(xr).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
