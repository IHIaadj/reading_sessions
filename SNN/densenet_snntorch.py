from collections import OrderedDict
from typing import Any, List, Tuple
import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
num_steps = 100
spike_grad = surrogate.fast_sigmoid()

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 norm_layer: callable, bias: bool, node: callable, **kwargs):
        super().__init__()
        
        self.drop_rate = float(drop_rate)
        output_c =  bn_size * growth_rate
        self.layer = nn.Sequential(
            norm_layer(num_input_features),
            nn.Conv2d(num_input_features,output_c, kernel_size=1, stride=1, bias=bias),
            snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
            norm_layer(output_c),
            nn.Conv2d(output_c, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias),
            snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        x = torch.cat(prev_features, 1)
        out = self.layer(x)
        return out

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, 
                 growth_rate: int, drop_rate: float, norm_layer: callable, bias: bool, 
                 node: callable = None, **kwargs):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
                node=node,
                bias=bias,
                **kwargs
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, 
                 norm_layer: callable, bias: bool, node: callable = None, **kwargs):
        super().__init__()
        self.add_module("normt", norm_layer(num_input_features))
        self.add_module("convt", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=bias))
        self.add_module("actt", snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True))
        self.add_module("poolt", nn.AvgPool2d(kernel_size=2, stride=2))

class SpikingDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_channels=2, bn_size=4, drop_rate=0, 
                 num_classes=10, init_weights=True, norm_layer: callable = None, 
                 node: callable = None, **kwargs):
        
        super().__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        num_init_features = 2 * growth_rate
        #print(bias)
        # First convolution
        #self.pad0 = nn.ConstantPad2d(1, 0.)
        #self.norm0 = norm_layer(num_init_channels)
        #self.conv0 = nn.Conv2d(num_init_channels, num_init_features, 
        #                                kernel_size=3, stride=2, padding=0, bias=bias)
        #self.act0 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        #self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("act0", snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
                bias=bias,
                node=node,
                **kwargs
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            
            # register feature maps size after trans1, trans2, dense4 (not after trans3) for OD
            if i != len(block_config) - 2:
                self.out_channels.append(num_features)
                
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, 
                    num_output_features=num_features // 2, 
                    norm_layer=norm_layer,
                    bias=bias,
                    node=node,
                    **kwargs
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2
     
        self.classifier = nn.Linear(num_features, num_classes)
        self.class_act = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        #features = self.forward_features(x)
        features = self.features(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        out, mem = self.class_act(out)
        #out = out.flatten(start_dim=-2).sum(dim=-1)
        return out, mem #final leaky membrane potential 
    

def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_channels: int,
    norm_layer: callable = None, single_step_neuron: callable = None,
    **kwargs: Any,
) -> SpikingDenseNet:
    model = SpikingDenseNet(growth_rate, block_config, num_init_channels, norm_layer=norm_layer, node=single_step_neuron, **kwargs)
    return model

def spiking_densenet_custom(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs) -> SpikingDenseNet:
    r"""A spiking version of custom DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet", growth_rate, block_config, num_init_channels, norm_layer, single_step_neuron, **kwargs)

def spiking_densenet121(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet121", 32, (2, 3, 8, 4), num_init_channels, norm_layer, single_step_neuron, **kwargs)

if __name__ == "__main__":
    tau = 2.0
    num_steps = 100
    #single_step_neuron = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
    model = spiking_densenet121(10)
    print(model)