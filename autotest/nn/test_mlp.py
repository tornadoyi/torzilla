import unittest
import torch
from torzilla import nn
    
class TestMLP(unittest.TestCase):
    
    def test_nn_layer(self):
        N, D, O = 128, 30, 10
        x = torch.rand([N, D])
        layer = nn.Layer(
            D,
            O,
            activation=nn.ReLU(),
            layer_norm=dict(eps=1e-3, elementwise_affine=True),
            batch_norm=dict(momentum=0.3, track_running_stats=False)
        )
        y = layer(x)
        self.assertEqual(y.shape, (128, 10))

    def test_nn_mlp_dft(self):
        N, D, O = 128, 30, 10
        x = torch.rand([N, D])
        mlp = nn.MLP(D, [20, 15, O])
        y = mlp(x)
        self.assertEqual(y.shape, (128, 10))

    def test_nn_mlp_diy(self):
        N, D, O = 128, 30, 10
        x = torch.rand([N, D])
        mlp = nn.MLP(
            D, 
            [20, 15, O],
            activations=[nn.ReLU()] * 3,
            layer_norms=[{}] * 3,
            batch_norms=[{}] * 3,
        )
        y = mlp(x)
        self.assertEqual(y.shape, (128, 10))

if __name__ == '__main__':
    unittest.main()