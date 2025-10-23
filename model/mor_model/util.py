from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.utils import ModelOutput


class LinearRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
    def forward(self, x):
        return self.router(x)
    
    
class MLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)
    
    
class DeepMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        
        # LayerNorm for input stabilization
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-5)
        
        # MLP router
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, out_dim, bias=True)
        )
        
        # Learnable depth bias for residual connection
        self.depth_bias = nn.Parameter(torch.zeros(out_dim))
        
        # Initialize MLP weights
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        # LayerNorm -> MLP -> Add residual bias
        h = self.ln(x)
        return self.router(h) + self.depth_bias

class DeepWideMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        
        # LayerNorm for input stabilization
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-5)
        
        # MLP router
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=True)
        )
        
        # Learnable depth bias for residual connection
        self.depth_bias = nn.Parameter(torch.zeros(out_dim))
        
        # Initialize MLP weights
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        # LayerNorm -> MLP -> Add residual bias
        h = self.ln(x)
        return self.router(h) + self.depth_bias

class TuneMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        
        # LayerNorm for input stabilization
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-5)
        
        # MLP router
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, out_dim, bias=True)
        )
        
        # Learnable depth bias for residual connection
        self.depth_bias = nn.Parameter(torch.zeros(out_dim))
        
        # Initialize MLP weights
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        # LayerNorm -> MLP -> Add residual bias
        h = self.ln(x)
        return self.router(h) + self.depth_bias
    
    
ROUTER_TYPES = {
    "linear": LinearRouter, 
    "mlp": MLPRouter, 
    "mlp_deep": DeepMLPRouter,
    "mlp_deep_wide": DeepWideMLPRouter,
    "mlp_tune": TuneMLPRouter,
}


@dataclass
class MoRLayerOutputWithPast(ModelOutput):

    hidden_state: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None
    selected_tokens: Optional[torch.FloatTensor] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None