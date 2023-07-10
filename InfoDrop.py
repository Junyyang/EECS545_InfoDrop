# EECS 545 Final Project
# Implementation Track
# Infomation Dropout

import torch
from torch import nn
from torch.nn.modules import padding

class InfoDrop(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, other_layers: list, radius: int = 3, neighbor_sample_num=None, drop_rate=1.5, temperature=0.03, bandwidth=1.0):
        """
        Construct the InfoDrop concatenated after the given layers.
        Based on the definition of InfoDrop, the module must start with a 
        nn.Conv2d layer.

        InfoDrop plugs into the module as the following sequence:
        Conv2d -> (other_layers, e.g. ReLU(), bn()) -> InfoDrop()

        Parameters will be extracted from the convolutional layers as many as possible.

        Parameters:
            - conv_layer: (nn.Conv2d) the convolutional layer.
            - other_layers: (list of (nn.Module)) a list consisting of the rest of the 
              layers.
            - radius: (int) the radius of the neighborhood.
            - neighbor_sample_num: (int or None) the num of samples from the neighbor
              of a patch. If None or greater than (2R + 1)^2, then all neighbors will 
              be used.
            - drop_rate: (float) the basic dropout rate r_0.
            - temperature: (float) the temperature in the Boltzmann distribution.
            - bandwidth: (float) the bandwidth h in the Gaussian kernel.
        """
        super(InfoDrop, self).__init__()
        self.conv = conv_layer
        self.others = other_layers
        self.R = radius
        self.dr = drop_rate
        self.T = temperature
        self.h = bandwidth
        if not neighbor_sample_num:
            self.nsn = (2 * self.R + 1) ** 2
        else:
            self.nsn = min(neighbor_sample_num, (2 * self.R + 1) ** 2)
        
        # Extracting necessary params from Conv2d
        self.stride = self.conv.stride[0]
        self.padding = self.conv.padding[0]
        self.kernel_size = self.conv.kernel_size[0]
        self.indim = self.conv.in_channels
        self.outdim = self.conv.out_channels

        # This will be changed to False during finetuning time or evaluation time
        self.infodrop_state = True

        # Adding one extra (row + column) at the end for later convenience
        self.padder = nn.ConstantPad2d(
            (self.padding + self.R,
            self.padding + self.R + 1,
            self.padding + self.R,
            self.padding + self.R + 1),
            0)

        self.find_patch_conv = nn.Conv2d(self.nsn, self.nsn, kernel_size=self.kernel_size, stride=self.stride, bias=False, groups=self.nsn)
        self.find_patch_conv.weight.data = torch.ones_like(self.find_patch_conv.weight, dtype=torch.float)
        self.find_patch_conv.weight.requires_grad = False
        self.sum_neighbor_conv = nn.Conv2d(self.nsn, self.outdim, kernel_size=1, padding=0, bias=False)
        self.sum_neighbor_conv.weight.data = torch.ones_like(self.sum_neighbor_conv.weight, dtype=torch.float)
        self.sum_neighbor_conv.weight.requires_grad = False

    def eval(self):
        self.infodrop_state = False
    
    def train(self):
        self.infodrop_state = True

    def random_sample(self, probability, count):
        bs, c, h, w = probability.shape
        return torch.multinomial((probability.view(bs * c, -1) + 1e-8), count, replacement=True)
    
    def forward(self, x):
        if not self.infodrop_state:
            x = self.conv(x)
            for l in self.others:
                x = l(x)
            return x

        x_clone = x.clone()
        x = self.conv(x)
        for l in self.others:
            x = l(x)

        with torch.no_grad():
            padded_x = self.padder(x_clone)
            origin_x = padded_x[:, :, self.R:-self.R-1, self.R:-self.R-1]
            if self.nsn:
                x_sample = (torch.randint(self.R * 2 + 1, size=(self.nsn,)) - self.R).tolist()
                y_sample = (torch.randint(self.R * 2 + 1, size=(self.nsn,)) - self.R).tolist()
                coor = zip(x_sample, y_sample)
            else:
                coor = [(i, j) for i in range(-self.R, self.R + 1) for j in range(-self.R, self.R + 1)]
            diff_list = []
            for i, j in coor:
                shifted_x = padded_x[:, :, (self.R+i):(-self.R-1+i), (self.R+j):(-self.R-1+j)]
                diff = (shifted_x - origin_x).clone()
                diff_list.append(diff)
            diffs = torch.cat(diff_list, dim=1)
            bs, _, h, w = diffs.shape
            diffs = (diffs ** 2).view(-1, self.indim, h, w).sum(dim=1).view(bs, -1, h, w)
            diffs = self.find_patch_conv(diffs)
            diffs = torch.exp(
                -diffs / diffs.mean() / 2 / self.h ** 2
            )

            prob = (self.sum_neighbor_conv(diffs) / self.nsn) ** (1 / self.T)
            prob /= prob.sum(dim=(-2, -1), keepdim=True)

            bs, indim, h, w = x.shape
            choices = self.random_sample(prob, count=int(self.dr * h * w))
            mask = torch.ones((bs * indim, h * w), device=x.device)
            mask[torch.arange(bs * indim, device=x.device).view(-1, 1), choices] = 0
        
        return x * mask.view(x.shape)



if __name__ == '__main__':
    # A quick example on how to use our InfoDrop module
    import torchvision.models as models
    alex = models.alexnet()
    infolex_features = nn.Sequential(
        InfoDrop(alex.features[0], alex.features[1:2]),
        *alex.features[2:]
    )
    alex.features = infolex_features

    alex.cuda(0)
    r = torch.randn((2, 3, 224, 224)).cuda(0)
    print(alex(r))
