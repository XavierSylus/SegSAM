"""
实验记录与监控模块
支持 WandB 和 TensorBoard 日志记录
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Logger:
    """
    统一的日志记录器接口
    支持 WandB 和 TensorBoard
    """
    
    def __init__(
        self,
        log_type: str = "tensorboard",  # "wandb", "tensorboard", "both", "none"
        experiment_name: Optional[str] = None,
        project_name: str = "SegSAM",
        log_dir: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            log_type: 日志类型 ("wandb", "tensorboard", "both", "none")
            experiment_name: 实验名称（如果为 None 则自动生成）
            project_name: 项目名称（WandB 使用）
            log_dir: 日志目录（TensorBoard 使用）
            wandb_entity: WandB entity（用户名或团队名）
            config: 配置字典（用于记录超参数）
        """
        self.log_type = log_type
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.project_name = project_name
        self.log_dir = log_dir
        self.config = config or {}
        
        # 初始化 WandB
        self.wandb_run = None
        if log_type in ["wandb", "both"]:
            try:
                import wandb
                self.wandb = wandb
                
                # 初始化 WandB run
                wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    entity=wandb_entity,
                    config=self.config,
                    reinit=True
                )
                self.wandb_run = wandb.run
                print(f"✓ WandB 已初始化: {wandb.run.url if wandb.run else 'N/A'}")
            except ImportError:
                print("⚠ WandB 未安装，跳过 WandB 日志记录")
                print("  安装命令: pip install wandb")
                self.log_type = "tensorboard" if log_type == "both" else "none"
            except Exception as e:
                print(f"⚠ WandB 初始化失败: {e}")
                self.log_type = "tensorboard" if log_type == "both" else "none"
                self.wandb = None
                self.wandb_run = None
        
        # 初始化 TensorBoard
        self.tb_writer = None
        if log_type in ["tensorboard", "both"]:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.SummaryWriter = SummaryWriter
                
                # 创建日志目录
                if log_dir is None:
                    log_dir = Path("logs") / "tensorboard" / self.experiment_name
                else:
                    log_dir = Path(log_dir) / "tensorboard" / self.experiment_name
                
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(str(log_dir))
                print(f"✓ TensorBoard 已初始化: {log_dir}")
            except ImportError:
                print("⚠ TensorBoard 未安装，跳过 TensorBoard 日志记录")
                print("  安装命令: pip install tensorboard")
                if self.log_type == "tensorboard":
                    self.log_type = "none"
                self.tb_writer = None
            except Exception as e:
                print(f"⚠ TensorBoard 初始化失败: {e}")
                if self.log_type == "tensorboard":
                    self.log_type = "none"
                self.tb_writer = None
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称（基于时间戳）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experiment_{timestamp}"
    
    def log(self, metrics: Dict[str, float], step: int):
        """
        记录指标
        
        Args:
            metrics: 指标字典，例如 {'train_loss': 0.5, 'val_dice': 0.8}
            step: 步数（通常是轮数）
        """
        # WandB 记录
        if self.wandb_run is not None:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                print(f"⚠ WandB 记录失败: {e}")
        
        # TensorBoard 记录
        if self.tb_writer is not None:
            try:
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(key, value, step)
            except Exception as e:
                print(f"⚠ TensorBoard 记录失败: {e}")
    
    def log_scalar(self, tag: str, scalar_value: float, step: int):
        """
        记录单个标量值
        
        Args:
            tag: 标签名称
            scalar_value: 标量值
            step: 步数
        """
        self.log({tag: scalar_value}, step)
    
    def log_summary(self, summary: Dict[str, Any]):
        """
        记录总结性指标（最终结果）
        
        Args:
            summary: 总结字典
        """
        # WandB 记录
        if self.wandb_run is not None:
            try:
                self.wandb.run.summary.update(summary)
            except Exception as e:
                print(f"⚠ WandB 总结记录失败: {e}")
        
        # TensorBoard 记录（作为文本）
        if self.tb_writer is not None:
            try:
                summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
                self.tb_writer.add_text("summary", summary_text, 0)
            except Exception as e:
                print(f"⚠ TensorBoard 总结记录失败: {e}")
    
    def close(self):
        """关闭日志记录器"""
        if self.wandb_run is not None:
            try:
                self.wandb.finish()
                print("✓ WandB 已关闭")
            except Exception as e:
                print(f"⚠ WandB 关闭失败: {e}")
        
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
                print("✓ TensorBoard 已关闭")
            except Exception as e:
                print(f"⚠ TensorBoard 关闭失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


def create_logger(
    log_type: str = "tensorboard",
    experiment_name: Optional[str] = None,
    project_name: str = "SegSAM",
    log_dir: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Logger:
    """
    便捷函数：创建日志记录器
    
    Args:
        log_type: 日志类型 ("wandb", "tensorboard", "both", "none")
        experiment_name: 实验名称
        project_name: 项目名称
        log_dir: 日志目录
        wandb_entity: WandB entity
        config: 配置字典
    
    Returns:
        Logger 对象
    """
    return Logger(
        log_type=log_type,
        experiment_name=experiment_name,
        project_name=project_name,
        log_dir=log_dir,
        wandb_entity=wandb_entity,
        config=config
    )


if __name__ == "__main__":
    # 测试日志记录器
    print("=" * 60)
    print("测试日志记录器")
    print("=" * 60)
    
    # 测试 TensorBoard
    print("\n1. 测试 TensorBoard 日志记录器")
    logger_tb = create_logger(
        log_type="tensorboard",
        experiment_name="test_tensorboard",
        log_dir="logs"
    )
    
    # 记录一些测试数据
    for step in range(10):
        logger_tb.log({
            'test_loss': 1.0 / (step + 1),
            'test_accuracy': step * 0.1
        }, step=step)
    
    logger_tb.log_summary({'final_loss': 0.1, 'final_accuracy': 1.0})
    logger_tb.close()
    print("✓ TensorBoard 测试完成")
    
    # 测试 WandB（如果可用）
    print("\n2. 测试 WandB 日志记录器（如果可用）")
    try:
        logger_wb = create_logger(
            log_type="wandb",
            experiment_name="test_wandb",
            project_name="test-project"
        )
        
        for step in range(5):
            logger_wb.log({
                'test_loss': 1.0 / (step + 1),
                'test_accuracy': step * 0.1
            }, step=step)
        
        logger_wb.log_summary({'final_loss': 0.1, 'final_accuracy': 1.0})
        logger_wb.close()
        print("✓ WandB 测试完成")
    except Exception as e:
        print(f"⚠ WandB 测试跳过: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n查看 TensorBoard:")
    print("  tensorboard --logdir logs/tensorboard")


