"""
适配器模块
定义 Adapter 类：Linear -> Activation -> Linear
用于参数高效的模型适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Type


class Adapter(nn.Module):
    """
    适配器模块
    架构: Linear -> Activation -> Linear
    
    用于参数高效的模型适配，通常插入到预训练模型中，
    只训练适配器参数，而冻结主模型参数。
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        adapter_dim: int = 64,
        activation: Union[str, Type[nn.Module], nn.Module] = "relu",
        dropout: float = 0.0,
        use_residual: bool = True,
        init_scale: float = 1e-3,
    ):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度（如果为 None，则等于 in_dim）
            adapter_dim: 适配器内部维度（瓶颈维度，通常远小于 in_dim）
            activation: 激活函数
                - str: "relu", "gelu", "tanh", "sigmoid", "swish"
                - Type[nn.Module]: 激活函数类
                - nn.Module: 激活函数实例
            dropout: Dropout 概率（默认: 0.0）
            use_residual: 是否使用残差连接（默认: True）
            init_scale: 初始化缩放因子（用于稳定训练，默认: 1e-3）
        """
        super(Adapter, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.adapter_dim = adapter_dim
        self.use_residual = use_residual and (in_dim == self.out_dim)
        
        # 第一个线性层：降维 (in_dim -> adapter_dim)
        self.down_proj = nn.Linear(in_dim, adapter_dim, bias=False)
        
        # 激活函数
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "sigmoid":
                self.activation = nn.Sigmoid()
            elif activation == "swish" or activation == "silu":
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        elif isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation
        
        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # 第二个线性层：升维 (adapter_dim -> out_dim)
        self.up_proj = nn.Linear(adapter_dim, self.out_dim, bias=False)
        
        # 初始化
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale: float) -> None:
        """初始化权重"""
        # 下投影层：使用较小的初始化
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        
        # 上投影层：初始化为零（这样初始时适配器输出接近零，残差连接更稳定）
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - Universal Adapter (支持3D序列和4D空间张量)
        
        Args:
            x: 输入张量
                - 3D序列: (B, N, in_dim) - Transformer输出
                - 4D空间: (B, C, H, W) 或 (B, H, W, C) - CNN特征图或ViT输出
                - 2D向量: (B, in_dim) - 全局特征
        
        Returns:
            输出张量，形状与输入相同（通道维度从in_dim变为out_dim）
        """
        # 处理不同的输入形状
        original_dim = x.dim()
        original_shape = x.shape
        need_permute_back = False
        
        if x.dim() == 4:
            # 4D空间张量: 需要判断是 (B, C, H, W) 还是 (B, H, W, C) 格式
            # 策略: 检查最后一维是否等于 in_dim
            if x.shape[-1] == self.in_dim:
                # 已经是 (B, H, W, C) 格式,无需 permute (SAM3 ViT 输出)
                pass
            elif x.shape[1] == self.in_dim:
                # (B, C, H, W) 格式,需要 permute
                x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
                need_permute_back = True
            else:
                # 无法确定格式,尝试使用最后一维
                print(f"[Warning] Ambiguous 4D input shape {original_shape}, in_dim={self.in_dim}")
                print(f"  Assuming last dimension is the channel dimension")
        elif x.dim() == 3:
            # 3D序列: (B, N, in_dim) - 已经是正确格式
            pass
        elif x.dim() == 2:
            # 2D向量: (B, in_dim) -> (B, 1, in_dim)
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}，期望2D/3D/4D")
        
        # 保存残差（在形状变换后，确保维度匹配）
        residual = x if self.use_residual else None
        
        # 适配器前向传播: Linear -> Activation -> Dropout -> Linear
        x = self.down_proj(x)      # (..., in_dim) -> (..., adapter_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)        # (..., adapter_dim) -> (..., out_dim)
        
        # 残差连接
        if self.use_residual:
            x = x + residual
        
        # 恢复原始形状
        if original_dim == 4 and need_permute_back:
            # (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        elif original_dim == 2:
            # (B, 1, out_dim) -> (B, out_dim)
            x = x.squeeze(1)
        
        return x
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"Adapter("
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"adapter_dim={self.adapter_dim}, "
            f"activation={self.activation}, "
            f"residual={self.use_residual}, "
            f"params={self.get_num_params()}"
            f")"
        )


class ParallelAdapter(nn.Module):
    """
    并行适配器
    将适配器与主路径并行，输出为 adapter(x) + x
    """
    
    def __init__(
        self,
        dim: int,
        adapter_dim: int = 64,
        activation: Union[str, Type[nn.Module], nn.Module] = "relu",
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ):
        """
        Args:
            dim: 输入/输出维度
            adapter_dim: 适配器内部维度
            activation: 激活函数
            dropout: Dropout 概率
            init_scale: 初始化缩放因子
        """
        super(ParallelAdapter, self).__init__()
        self.adapter = Adapter(
            in_dim=dim,
            out_dim=dim,
            adapter_dim=adapter_dim,
            activation=activation,
            dropout=dropout,
            use_residual=False,  # 在这里手动添加残差
            init_scale=init_scale,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：adapter(x) + x"""
        return self.adapter(x) + x


class SequentialAdapter(nn.Module):
    """
    序列适配器
    将适配器串联在主路径中，输出为 adapter(x)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adapter_dim: int = 64,
        activation: Union[str, Type[nn.Module], nn.Module] = "relu",
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            adapter_dim: 适配器内部维度
            activation: 激活函数
            dropout: Dropout 概率
            init_scale: 初始化缩放因子
        """
        super(SequentialAdapter, self).__init__()
        self.adapter = Adapter(
            in_dim=in_dim,
            out_dim=out_dim,
            adapter_dim=adapter_dim,
            activation=activation,
            dropout=dropout,
            use_residual=False,
            init_scale=init_scale,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：adapter(x)"""
        return self.adapter(x)


def create_adapter(
    in_dim: int,
    out_dim: Optional[int] = None,
    adapter_dim: int = 64,
    activation: str = "relu",
    **kwargs
) -> Adapter:
    """
    便捷函数：创建适配器
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        adapter_dim: 适配器内部维度
        activation: 激活函数名称
        **kwargs: 其他参数传递给 Adapter
    
    Returns:
        Adapter 实例
    """
    return Adapter(
        in_dim=in_dim,
        out_dim=out_dim,
        adapter_dim=adapter_dim,
        activation=activation,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Adapter 测试")
    print("=" * 60)
    
    # 测试基本适配器
    print("\n1. 测试基本适配器（带残差连接）")
    adapter = Adapter(
        in_dim=768,
        adapter_dim=64,
        activation="relu",
        use_residual=True
    )
    
    x = torch.randn(2, 196, 768)  # (B, N, dim)
    y = adapter(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数数量: {adapter.get_num_params():,}")
    print(f"适配器: {adapter}")
    
    # 测试不同激活函数
    print("\n2. 测试不同激活函数")
    for act in ["relu", "gelu", "tanh", "swish"]:
        adapter = Adapter(768, adapter_dim=64, activation=act)
        y = adapter(x)
        print(f"  {act}: 输出形状 {y.shape}, 参数 {adapter.get_num_params():,}")
    
    # 测试不同输入形状
    print("\n3. 测试不同输入形状")
    test_cases = [
        (torch.randn(2, 768), "2D: (B, dim)"),
        (torch.randn(2, 196, 768), "3D: (B, N, dim)"),
        (torch.randn(2, 768, 14, 14), "4D: (B, C, H, W)"),
    ]
    
    for x_test, desc in test_cases:
        adapter = Adapter(in_dim=x_test.shape[-1], adapter_dim=64)
        y_test = adapter(x_test)
        print(f"  {desc}: {x_test.shape} -> {y_test.shape}")
    
    # 测试并行适配器
    print("\n4. 测试并行适配器")
    parallel_adapter = ParallelAdapter(dim=768, adapter_dim=64)
    x_parallel = torch.randn(2, 196, 768)
    y_parallel = parallel_adapter(x_parallel)
    print(f"输入形状: {x_parallel.shape}")
    print(f"输出形状: {y_parallel.shape}")
    print(f"残差连接: {torch.allclose(y_parallel, parallel_adapter.adapter(x_parallel) + x_parallel)}")
    
    # 测试序列适配器
    print("\n5. 测试序列适配器（不同维度）")
    seq_adapter = SequentialAdapter(in_dim=768, out_dim=512, adapter_dim=64)
    x_seq = torch.randn(2, 196, 768)
    y_seq = seq_adapter(x_seq)
    print(f"输入形状: {x_seq.shape}")
    print(f"输出形状: {y_seq.shape}")
    
    # 测试 Dropout
    print("\n6. 测试 Dropout")
    adapter_dropout = Adapter(768, adapter_dim=64, dropout=0.1)
    y_dropout = adapter_dropout(x)
    print(f"带 Dropout 的输出形状: {y_dropout.shape}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

