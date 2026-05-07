"""
Agent 配置加载器
从 YAML 文件加载配置并创建 Agent 实例
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml
except ImportError:
    yaml = None
    print("警告: PyYAML 未安装，配置加载功能将不可用。请运行: pip install pyyaml")

from src.agent_inference import create_agent, MedicalImageAgent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None，使用默认路径
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 使用默认配置文件路径
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "agent_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if yaml is None:
        raise ImportError("需要安装 PyYAML 才能加载配置文件: pip install pyyaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def create_agent_from_config(
    config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None
) -> MedicalImageAgent:
    """
    从配置文件创建 Agent 实例
    
    Args:
        config_path: 配置文件路径
        override_config: 覆盖配置的字典（优先级高于文件配置）
    
    Returns:
        MedicalImageAgent 实例
    """
    # 加载配置
    config = load_config(config_path)
    
    # 应用覆盖配置
    if override_config:
        # 深度合并配置
        def deep_merge(base: dict, override: dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, override_config)
    
    # 提取配置
    sam3_config = config.get("sam3", {})
    llm_config = config.get("llm", {})
    output_config = config.get("output", {})
    
    # 创建 Agent
    agent = create_agent(
        sam3_model_path=sam3_config.get("model_path"),
        llm_server_url=llm_config.get("server_url"),
        llm_model=llm_config.get("model"),
        llm_api_key=llm_config.get("api_key"),
        output_dir=output_config.get("output_dir", "agent_output"),
        device=sam3_config.get("device", "cuda"),
        debug=output_config.get("debug", False),
    )
    
    return agent


def get_inference_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取推理配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        推理配置字典
    """
    config = load_config(config_path)
    inference_config = config.get("inference", {})
    medical_config = config.get("medical_image", {})
    
    return {
        "max_generations": inference_config.get("max_generations", 100),
        "cleanup_temp_files": inference_config.get("cleanup_temp_files", True),
        "default_slice_idx": medical_config.get("default_slice_idx"),
        "output_format": medical_config.get("output_format", "png"),
    }


if __name__ == "__main__":
    # 示例：从配置文件创建 Agent
    try:
        agent = create_agent_from_config()
        print("✓ Agent 创建成功")
        print(f"  输出目录: {agent.output_dir}")
        print(f"  设备: {agent.device}")
        print(f"  调试模式: {agent.debug}")
    except FileNotFoundError as e:
        print(f"✗ 配置文件未找到: {e}")
        print("请确保 config/agent_config.yaml 存在")
    except Exception as e:
        print(f"✗ 创建 Agent 失败: {e}")
        import traceback
        traceback.print_exc()

