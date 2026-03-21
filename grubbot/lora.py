import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Custom LoRA module to replace nn.Linear
    W'x = Wx + B(Ax)
    """
    def __init__(self, in_features: int, out_features: int, r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0.05, base_weight=None, base_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        self.base_layer = nn.Linear(in_features, out_features, bias=(base_bias is not None))
        if base_weight is not None:
            self.base_layer.weight.data.copy_(base_weight)
        if base_bias is not None:
            self.base_layer.bias.data.copy_(base_bias)
            
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(self.base_layer.weight.new_empty((r, in_features)))
            self.lora_B = nn.Parameter(self.base_layer.weight.new_empty((out_features, r)))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r > 0:
            lo_x = self.dropout(x)
            lora_update = (lo_x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result += lora_update
        return result

def patch_model_with_lora(model: nn.Module, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], r=16, lora_alpha=16):
    """
    Recursively replace target Linear layers with LoRALinear
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            lora_layer = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                base_weight=module.weight.data,
                base_bias=module.bias.data if module.bias is not None else None
            )
            setattr(model, name, lora_layer)
        else:
            patch_model_with_lora(module, target_modules, r, lora_alpha)
            
    # Keep only LoRA parameters trainable
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    
    return model
