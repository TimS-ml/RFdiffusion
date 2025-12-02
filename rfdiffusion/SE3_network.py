"""
SE(3)-equivariant neural networks for geometric deep learning on protein structures.

This module implements SE(3)-equivariant transformers that process 3D geometric data
while respecting rotational and translational symmetries. These networks are used to
update structural features in a rotation-equivariant manner, which is critical for
protein structure prediction tasks.
"""
import torch
import torch.nn as nn

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from rfdiffusion.util_module import init_lecun_normal_param
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber

class SE3TransformerWrapper(nn.Module):
    """
    Wrapper for SE(3)-equivariant Graph Convolutional Network with attention.

    This class wraps the SE3Transformer to provide an SE(3)-equivariant network
    that can process geometric features of protein structures. The network maintains
    equivariance to 3D rotations and translations, making it suitable for structural
    biology applications.
    """
    def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=3, l1_out_features=2,
                 num_edge_features=32):
        """
        Initialize the SE(3)-equivariant transformer.

        Args:
            num_layers (int): Number of transformer layers
            num_channels (int): Number of channels in hidden layers
            num_degrees (int): Maximum degree of spherical harmonics (controls feature types)
            n_heads (int): Number of attention heads
            div (int): Channel division factor for multi-head attention
            l0_in_features (int): Number of type-0 (scalar) input features
            l0_out_features (int): Number of type-0 (scalar) output features
            l1_in_features (int): Number of type-1 (vector) input features
            l1_out_features (int): Number of type-1 (vector) output features
            num_edge_features (int): Number of edge features
        """
        super().__init__()
        # Build the network
        self.l1_in = l1_in_features
        #
        fiber_edge = Fiber({0: num_edge_features})
        if l1_out_features > 0:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
        else:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})
        
        self.se3 = SE3Transformer(num_layers=num_layers,
                                  fiber_in=fiber_in,
                                  fiber_hidden=fiber_hidden,
                                  fiber_out = fiber_out,
                                  num_heads=n_heads,
                                  channels_div=div,
                                  fiber_edge=fiber_edge,
                                  use_layer_norm=True)
                                  #use_layer_norm=False)

        self.reset_parameter()

    def reset_parameter(self):
        """
        Initialize network parameters using appropriate initialization schemes.

        Uses LeCun normal initialization for most layers and Kaiming normal for
        radial function layers. The final layer is zero-initialized to ensure
        stable training at the start.
        """

        # make sure linear layer before ReLu are initialized with kaiming_normal_
        for n, p in self.se3.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            elif len(p.shape) == 1:
                continue
            else:
                if "radial_func" not in n:
                    p = init_lecun_normal_param(p) 
                else:
                    if "net.6" in n:
                        nn.init.zeros_(p)
                    else:
                        nn.init.kaiming_normal_(p, nonlinearity='relu')
        
        # make last layers to be zero-initialized
        #self.se3.graph_modules[-1].to_kernel_self['0'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['0'])
        #self.se3.graph_modules[-1].to_kernel_self['1'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['1'])
        nn.init.zeros_(self.se3.graph_modules[-1].to_kernel_self['0'])
        nn.init.zeros_(self.se3.graph_modules[-1].to_kernel_self['1'])

    def forward(self, G, type_0_features, type_1_features=None, edge_features=None):
        """
        Forward pass through the SE(3)-equivariant network.

        Args:
            G: Graph structure containing node and edge connectivity
            type_0_features (torch.Tensor): Scalar (type-0) node features
            type_1_features (torch.Tensor, optional): Vector (type-1) node features
            edge_features (torch.Tensor, optional): Edge features

        Returns:
            dict: Dictionary containing output features organized by type
                  (e.g., {'0': scalar_features, '1': vector_features})
        """
        if self.l1_in > 0:
            node_features = {'0': type_0_features, '1': type_1_features}
        else:
            node_features = {'0': type_0_features}
        edge_features = {'0': edge_features}
        return self.se3(G, node_features, edge_features)
