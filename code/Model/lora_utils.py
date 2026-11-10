import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wrap a frozen nn.Linear with trainable LoRA adapters.
    Forward: y = base(x) + scale * B(A(dropout(x)))
    """

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got {type(base_layer)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = next(base_layer.parameters()).device
        self.lora_A = nn.Linear(self.in_features, r, bias=False).to(device)
        self.lora_B = nn.Linear(r, self.out_features, bias=False).to(device)
        self.reset_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    # PyTorch internals (such as MultiheadAttention) access .weight / .bias directly;
    # delegate these attributes to base_layer to avoid AttributeError
    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out



def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    if not module_name:
        raise ValueError("module_name must be non-empty")
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]



def _match_any_keyword(module_name: str, keywords: Sequence[str]) -> bool:
    return any(keyword and keyword in module_name for keyword in keywords)


def _inactive_parameter_ids(model: nn.Module):
    """Identify bypassed parameters without changing their original autograd flags."""
    inactive = set()
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention) and isinstance(module.out_proj, LoRALinear):
            # Native MHA reads base weight/bias and never calls LoRALinear.forward.
            inactive.update(id(p) for p in module.out_proj.lora_A.parameters())
            inactive.update(id(p) for p in module.out_proj.lora_B.parameters())
    if hasattr(model, 'inactive_classification_parameters'):
        inactive.update(id(p) for p in model.inactive_classification_parameters())
    return inactive



def apply_lora_to_model(
    model: nn.Module,
    target_keywords: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    verbose: bool = True,
) -> List[str]:
    """
    Replace matching nn.Linear modules with LoRALinear.
    Returns the list of replaced module names.
    """
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and _match_any_keyword(module_name, target_keywords):
            parent, child_name = _get_parent_module(model, module_name)
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(module_name)

    if verbose:
        print("=" * 80)
        print("LoRA modules injected:")
        if replaced:
            for name in replaced:
                parent, child_name = _get_parent_module(model, name)
                inactive = isinstance(parent, nn.MultiheadAttention) and child_name == 'out_proj'
                suffix = ' [inactive adapters excluded from optimizer; retained for compatibility]' if inactive else ''
                print(f"  - {name}{suffix}")
        else:
            print("  [Warning] no modules matched the LoRA target keywords.")
        print("=" * 80)
    return replaced



def mark_only_lora_and_selected_trainable(
    model: nn.Module,
    extra_trainable_keywords: Sequence[str] = ("projection",),
    train_bias: str = "none",
):
    """
    Freeze all params, then unfreeze:
      1) LoRA A/B params
      2) params whose names match extra_trainable_keywords
      3) biases if train_bias in {'all', 'lora_only'}
    """
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            continue

        if any(keyword and keyword in name for keyword in extra_trainable_keywords):
            param.requires_grad = True
            continue

        if train_bias == "all" and name.endswith("bias"):
            param.requires_grad = True
        elif train_bias == "lora_only" and name.endswith("bias") and ("base_layer" not in name):
            param.requires_grad = True


def extract_lora_state_dict(
    model: nn.Module,
    extra_trainable_keywords: Sequence[str] = ("projection",),
):
    state = {}
    for name, tensor in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            state[name] = tensor.detach().cpu()
            continue
        if any(keyword and keyword in name for keyword in extra_trainable_keywords):
            state[name] = tensor.detach().cpu()
    return state



def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in get_trainable_parameter_groups(model))
    return total, trainable



def print_trainable_parameters(model: nn.Module):
    total, trainable = count_parameters(model)
    ratio = 100.0 * trainable / max(total, 1)
    print("=" * 80)
    print(f"Total params     : {total:,}")
    print(f"Optimizer params : {trainable:,}")
    print(f"Optimizer ratio  : {ratio:.4f}%")
    print("=" * 80)



def get_trainable_parameter_groups(model: nn.Module) -> Iterable[nn.Parameter]:
    inactive = _inactive_parameter_ids(model)
    return (p for p in model.parameters() if p.requires_grad and id(p) not in inactive)
