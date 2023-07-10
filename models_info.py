from InfoDrop import InfoDrop
from torchvision.models import alexnet, resnet


def alexnet_infodrop():
    alex = alexnet()
    infolex_features = nn.Sequential(
        InfoDrop(alex.features[0], alex.features[1:2]),
        *alex.features[2:]
    )
    alex.features = infolex_features
    return alex

class BasicBlock_infodrop(resnet.BasicBlock):
    def __init__(self, *args, radius: int = 3, neighbor_sample_num=None, drop_rate=1.5, temperature=0.03, bandwidth=1.0, **kwargs):
        super(BasicBlock_infodrop, self).__init__(*args, **kwargs)
        self.infodrop = InfoDrop(self.conv1, [self.bn1, self.relu], radius=radius, neighbor_sample_num=neighbor_sample_num, drop_rate=drop_rate, temperature=temperature, bandwidth=bandwidth)

    def forward(self, x):
        identity = x

        out = self.infodrop(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet18_infodrop(**kwargs):
    return resnet.ResNet(BasicBlock_infodrop, [2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    print(resnet.resnet18())
    r = resnet18_infodrop()
    print(r)
