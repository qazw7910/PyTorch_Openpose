import torch
import torch.nn as nn
import torchvision
from basic_ops import ConsensusModule


class TSN(nn.Module):

    def __init__(self, num_class, num_segments, modality, base_model='resnet50', transforms=None,
                 new_length=1, partial_bn=True, dropout=0):
        super(TSN, self).__init__()

        self.num_class = num_class
        self.num_segments = num_segments
        self.modality = modality
        self.base_model = base_model
        self.transform = transforms
        self.new_length = new_length
        self.partial_bn = partial_bn
        self.consensus = ConsensusModule('avg')

        self._prepare_base_model()

        if self.modality == 'RGBdiff':
            self._construct_rgb_diff_model()

        '''
        params = self.base_model.named_parameters()
        for i, name_param in enumerate(params):
            name, param = name_param
            if i > 29:
                param.requires_grad = False
        '''
        fc_input_dim = getattr(self.base_model, self.last_layer_name).in_features

        if dropout > 0:
            last_layer = nn.Sequential(nn.Dropout(dropout), nn.Linear(fc_input_dim, num_class))
        else:
            last_layer = nn.Linear(fc_input_dim, num_class)

        setattr(self.base_model, self.last_layer_name, last_layer)

    def _prepare_base_model(self):
        if 'resnet' in self.base_model:
            self.base_model = getattr(torchvision.models, self.base_model)(pretrained=True)
            self.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

    def _construct_rgb_diff_model(self):
        modules = list(self.base_model.modules())

        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length, ) + kernel_size[2:]

        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels, conv_layer.kernel_size,
                             stride=conv_layer.stride, padding=conv_layer.padding,
                             bias=False if conv_layer.bias is None else True)

        with torch.no_grad():
            new_conv.weight.copy_(new_kernels)

            if conv_layer.bias is not None:
                new_conv.bias.copy_(params[1])

        conv_layer_name = list(container.state_dict().keys())[0][:-7]

        setattr(container, conv_layer_name, new_conv)

    def train(self, mode=True):
        # Override the default train() to freeze the BN parameters

        super(TSN, self).train(mode)
        '''
        cnt = 1
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if cnt >= (2 if self.partial_bn else 1):
                    m.eval()

                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

                cnt += 1
        '''

    def forward(self, input):
        out = self.base_model(input.view((-1, 3 * self.new_length) + input.size()[-2:]))
        out = out.view((-1, self.num_segments, self.num_class))
        out = self.consensus(out)
        return out.squeeze(1)


if __name__ == '__main__':
    model = TSN(num_class=4, num_segments=10, new_length=3,
                modality='RGBdiff', base_model='resnet18', dropout=0)
