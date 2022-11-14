from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
num_steps = 100
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 norm_layer: callable, bias: bool, node: callable, **kwargs):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.drop_rate = float(drop_rate)
        
        self.norm1 = norm_layer(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=bias)
        self.act1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=False)

        self.norm2 = norm_layer(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        global num_steps
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        x = torch.cat(prev_features, 1)
        
        # reset hidden states and outputs at t=0
        mem1 = self.act1.init_leaky()
        mem2 = self.act2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.conv1(self.norm1(x))
            spk1, mem1 = self.act1(cur1, mem1)
            cur2 = self.conv2(self.norm2(spk1))
            spk2, mem2 = self.act2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return spk2_rec[-1], mem2_rec[-1]

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
        print(init_features.shape)
        for name, layer in self.items():
            print(layer)
            new_features, _ = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, 
                 norm_layer: callable, bias: bool, node: callable = None, **kwargs):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.add_module("norm", norm_layer(num_input_features))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=bias))
        self.add_module("act", snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True))
        self.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2))

class SpikingDenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_channels=2, bn_size=4, drop_rate=0, 
                 num_classes=1000, init_weights=True, norm_layer: callable = None, 
                 node: callable = None, **kwargs):
        
        super().__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        num_init_features = 2 * growth_rate
        spike_grad = surrogate.fast_sigmoid()
        print(bias)
        # First convolution
        self.pad0 = nn.ConstantPad2d(1, 0.)
        self.norm0 = norm_layer(num_init_channels)
        self.conv0 = nn.Conv2d(num_init_channels, num_init_features, 
                                        kernel_size=3, stride=2, padding=0, bias=bias)
        self.act0 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=False)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential()
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
                
        self.norm_classif = norm_layer(num_features)
        self.conv_classif = nn.Conv2d(num_features, num_classes, 
                                                kernel_size=1, bias=bias)
        self.act_classif = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=False)


    def forward_classif(self, x): 
        global num_steps
        # reset hidden states and outputs at t=0
        mem1 = self.act_classif.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.conv_classif(self.norm_classif(x))
            spk1, mem1 = self.act_classif(cur1, mem1)

            spk2_rec.append(spk1)
            mem2_rec.append(mem1)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
    def forward_features(self, x):
        global num_steps
        # reset hidden states and outputs at t=0
        mem1 = self.act0.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.conv0(self.norm0((self.pad0(x))))
            spk1, mem1 = self.act0(cur1, mem1)
            cur2 = self.pool0(spk1)

            spk2_rec.append(cur2)
            mem2_rec.append(mem1)

        return spk2_rec[-1], mem2_rec[-1]
    
    def forward(self, x):
        print("start")
        features, _ = self.forward_features(x)
        print("done1")
        features,_ = self.features(features)
        print("done2")
        out, _ = self.forward_classif(features)
        out = out.flatten(start_dim=-2).sum(dim=-1)
        return out
    

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
    return _densenet("densenet121", 32, (6, 12, 24, 16), num_init_channels, norm_layer, single_step_neuron, **kwargs)

if __name__ == "__main__":
    tau = 2.0
    num_steps = 100
    #single_step_neuron = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
    model = spiking_densenet121(10)
    print(model)