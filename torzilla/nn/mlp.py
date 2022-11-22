from torch import nn as _nn
import numpy as np
from functools import partial

_WEIGHT_INITIALIZER = partial(_nn.init.orthogonal_, gain=np.sqrt(2))
_BIAS_INITIALIZER = partial(_nn.init.constant_, val=0)
_ACTIVATION = _nn.ReLU()

def get_layer_weight_initializer():
    return _WEIGHT_INITIALIZER

def set_layer_weight_initializer(initializer):
    global _WEIGHT_INITIALIZER
    _WEIGHT_INITIALIZER = initializer

def get_layer_bias_initializer():
    return _BIAS_INITIALIZER

def set_layer_bias_initializer(initializer):
    global _BIAS_INITIALIZER
    _BIAS_INITIALIZER = initializer

def get_mlp_activation():
    return _ACTIVATION

def set_mlp_activation(activation):
    global _ACTIVATION
    _ACTIVATION = activation


class Layer(_nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_initializer=get_layer_weight_initializer(),
        bias_initializer=get_layer_bias_initializer(),
        layer_norm=None,
        batch_norm=None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.ops = []
        
        # linear & initializer
        self.linear = _nn.Linear(in_features, out_features, bias, device, dtype)
        weight_initializer(self.linear.weight.data)
        if bias:
            bias_initializer(self.linear.bias.data)
        self.ops.append(self.linear)

        # layer norm
        if layer_norm is not None:
            self.layer_norm = _nn.LayerNorm(
                [out_features], 
                eps=layer_norm.get('eps', 1e-5), 
                elementwise_affine=layer_norm.get('elementwise_affine', True), 
                device=device, 
                dtype=dtype
            )
            self.ops.append(self.layer_norm)

        # batch norm
        if batch_norm is not None:
            self.batch_norm = _nn.BatchNorm1d(
                out_features,
                eps=batch_norm.get('eps', 1e-5),
                momentum=batch_norm.get('momentum', 0.1),
                affine=batch_norm.get('affine', True),
                track_running_stats=batch_norm.get('track_running_stats', True),
                device=device, 
                dtype=dtype
            )
            self.ops.append(self.batch_norm)

        # activation
        if activation is not None:
            self.activation = activation
            self.ops.append(self.activation)

    def forward(self, input):
        x = input
        for op in self.ops:
            x = op(x)
        return x


class MLP(_nn.Module):
    WEIGHT_INITIALIZER = partial(_nn.init.orthogonal, gain=np.sqrt(2))
    BIAS_INITIALIZER = partial(_nn.init.constant, val=0)

    def __init__(
        self,
        in_features,
        layer_features,
        bias=True,
        activations=get_mlp_activation(),
        weight_initializers=get_layer_weight_initializer(),
        bias_initializers=get_layer_bias_initializer(),
        layer_norms=None,
        batch_norms=None,
        device=None, 
        dtype=None
    ):
        super().__init__()

        # activation
        if not isinstance(activations, (tuple, list)):
            activations = [activations] * len(layer_features)

        # initializer
        if not isinstance(weight_initializers, (tuple, list)):
            weight_initializers = [weight_initializers] * len(layer_features)
        if not isinstance(bias_initializers, (tuple, list)):
            bias_initializers = [bias_initializers] * len(layer_features)

        # normalizer
        if not isinstance(layer_norms, (tuple, list)):
            layer_norms = [layer_norms] * len(layer_features)
        if not isinstance(batch_norms, (tuple, list)):
            batch_norms = [batch_norms] * len(layer_features)

        # layers
        layers = [None] * len(layer_features)
        last_features = in_features
        for i, out_feautures in enumerate(layer_features):
            layers[i] = Layer(
                last_features, 
                out_feautures, 
                bias=bias, 
                activation=activations[i], 
                weight_initializer=weight_initializers[i],
                bias_initializer=bias_initializers[i],
                layer_norm=layer_norms[i],
                batch_norm=batch_norms[i],
                device=device,
                dtype=dtype
            )
            last_features = out_feautures
        self.layers = _nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

