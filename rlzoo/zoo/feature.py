import numpy as np
import torch
from gym.spaces import Discrete, MultiDiscrete, Box
from torzilla import nn
from torzilla.nn import functional as F


__FEATURES__ = {}

def register(name):
    global __FEATURES__
    def _thunk(func):
        __FEATURES__[name] = func
        return func
    return _thunk


class CNN(nn.Module):
    def __init__(
        self, 
        space_shape, 
        out_channels, 
        kernel_sizes, 
        strides,
        out_features,
    ):
        super().__init__()

        if len(space_shape) == 2:
            in_channel, size = 1, space_shape
        elif len(space_shape) == 3:
            in_channel, size = space_shape[-1], space_shape[:-1]
        else:
            raise ValueError(f'cnn does not support space shape {space_shape}')

        layers = []
        for i in range(len(out_channels)):
            layers.append(nn.Conv2d(
                in_channel, 
                out_channels[i],
                kernel_sizes[i], 
                strides[i], 
                0
            ))
            in_channel = out_channels[i]
            size = nn.eval_conv_output_size(layers[-1], size)
            layers.append(nn.ReLU())
        
        layers.append(nn.Flatten())
        layers.append(nn.Layer(
            in_features = np.prod([in_channel] + list(size)),
            out_features=out_features,
        ))
        layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

        self.out_features = out_features

    def forward(self, input):
        x = input
        for l in self.layers:
            x = l(x)
        return x


@register('cnn_small')
class SmallCNN(CNN):
    def __init__(self, space_shape):
        super().__init__(
            space_shape=space_shape, 
            out_channels=[16, 32], 
            kernel_sizes=[(8, 8), (4, 4)], 
            strides=[(4, 4), (2, 2)], 
            out_features=256,
        )

@register('cnn_large')
class LargeCNN(CNN):
    def __init__(self, space_shape):
        super().__init__(
            space_shape=space_shape, 
            out_channels=[32, 64, 64], 
            kernel_sizes=[(8, 8), (4, 4), (3, 3)], 
            strides=[(4, 4), (2, 2), (1, 1)], 
            out_features=512
        )


@register('mlp')
class MLP(nn.MLP):
    def __init__(self, space_shape):

        if len(space_shape) != 1:
            raise ValueError(f'mlp does not support space shape {space_shape}')

        self.out_features = 64
        return super().__init__(
            in_features=space_shape[-1],
            out_features=[64, 64],
            activations=[nn.Tanh(), nn.Tanh()],
            layer_norms={}
        )


class Input(nn.Module):
    def __init__(self, ob_space):
        super().__init__()
        self._ob_space = ob_space
        self._out_shape = self._ob_space.shape
    
    @property
    def out_shape(self): 
        return self._out_shape

    def forward(self, input):
        return torch.as_tensor(input).to(torch.float32)
    

class DiscreteInput(Input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_shape = (self._ob_space.n, )

    def forward(self, input):
        return F.one_hot(
            torch.as_tensor(input), 
            self._ob_space.n
        ).to(torch.float32)


class MultiDiscreteInput(Input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_classes = self._ob_space.nvec.flatten().tolist()
        self._out_shape = (np.sum(self._num_classes), )
        
    def forward(self, input):
        input = torch.as_tensor(input)
        B = input.shape[0]
        input = input.reshape((B, -1))
        features = []
        for i, n in enumerate(self._num_classes):
            features.append(F.one_hot(input[:, i], n))
        return torch.cat(features, axis=-1)


class BoxInput(Input):
    pass


class ImageInput(BoxInput):
    def forward(self, input):
        return super().forward(input) / 255.0
        

class Feature(nn.Module):
    def __init__(self, ob_space) -> None:
        super().__init__()

        # input
        if isinstance(ob_space, Discrete):
            self.input = DiscreteInput(ob_space)
        elif isinstance(ob_space, MultiDiscrete):
            self.input = MultiDiscreteInput(ob_space)
        elif isinstance(ob_space, Box):
            if len(ob_space.shape) >= 2:
                self.input = ImageInput(ob_space)
            else:
                self.input = BoxInput(ob_space)
        else:
            ValueError(f'feature can not support ob_space {ob_space}')

        # extractor
        if isinstance(ob_space, Box) and len(ob_space.shape) >= 2:
            self.extractor = LargeCNN(self.input.out_shape)
        else:
            self.extractor = MLP(self.input.out_shape)

        self.out_features = self.extractor.out_features
        
    def forward(self, input):
        x = input
        x = self.input(x)
        x = self.extractor(x)
        return x
