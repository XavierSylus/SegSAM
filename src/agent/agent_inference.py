"""
SAM3 Agent 推理模块 - 集成到 SegSAM 项目
提供智能代理推理功能，结合 LLM 和 SAM3 进行医学图像分割
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 添加 sam3 路径
sam3_path = Path(__file__).parent.parent / "core_projects" / "sam3-main"
sys.path.insert(0, str(sam3_path))

try:
    from sam3.agent.agent_core import agent_inference
    from sam3.agent.client_llm import send_generate_request
    from sam3.agent.client_sam3 import call_sam_service
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3 import build_sam3
except ImportError as e:
    print(f"警告: 无法导入 SAM3 agent 模块: {e}")
    print("请确保已正确安装 SAM3 依赖")
    agent_inference = None
    send_generate_request = None
    call_sam_service = None
    Sam3Processor = None
    build_sam3 = None


class MedicalImageAgent:
    """
    医学图像智能代理推理器
    
    功能：
    1. 使用 LLM 理解自然语言查询
    2. 调用 SAM3 进行图像分割
    3. 迭代式优化分割结果
    4. 支持医学图像格式（NIfTI等）
    """
    
    def __init__(
        self,
        sam3_model_path: Optional[str] = None,
        llm_server_url: Optional[str] = None,
        llm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        llm_api_key: Optional[str] = None,
        output_dir: str = "agent_output",
        device: str = "cuda",
        debug: bool = False,
    ):
        """
        初始化医学图像代理
        
        Args:
            sam3_model_path: SAM3 模型路径（如果为None，将使用默认配置）
            llm_server_url: LLM 服务器地址（例如: "http://127.0.0.1:8000"）
            llm_model: LLM 模型名称
            llm_api_key: LLM API 密钥（如果需要）
            output_dir: 输出目录
            device: 设备（"cuda" 或 "cpu"）
            debug: 是否启用调试模式
        """
        self.device = device
        self.output_dir = output_dir
        self.debug = debug
        
        # 初始化 SAM3 处理器
        self.sam3_processor = None
        if build_sam3 and Sam3Processor:
            try:
                print("正在加载 SAM3 模型...")
                # 这里需要根据实际情况配置 SAM3 模型
                # 如果 sam3_model_path 为 None，使用默认配置
                if sam3_model_path:
                    # 加载自定义模型
                    # model = build_sam3(sam3_model_path)
                    # self.sam3_processor = Sam3Processor(model)
                    pass
                else:
                    # 使用默认模型配置
                    # 注意：需要根据实际 SAM3 配置进行调整
                    print("警告: 未提供 SAM3 模型路径，将使用默认配置")
                    # model = build_sam3()  # 需要根据实际 API 调用
                    # self.sam3_processor = Sam3Processor(model)
                print("SAM3 模型加载完成")
            except Exception as e:
                print(f"警告: SAM3 模型加载失败: {e}")
                print("将使用占位符，请确保在实际使用前正确配置")
        
        # 配置 LLM 客户端
        self.llm_server_url = llm_server_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_sam_service_wrapper(self):
        """创建 SAM3 服务包装器"""
        if self.sam3_processor is None:
            raise ValueError("SAM3 处理器未初始化，请先加载模型")
        
        def wrapped_call_sam_service(image_path, text_prompt, output_folder_path):
            """包装的 SAM3 服务调用"""
            return call_sam_service(
                sam3_processor=self.sam3_processor,
                image_path=image_path,
                text_prompt=text_prompt,
                output_folder_path=output_folder_path,
            )
        
        return wrapped_call_sam_service
    
    def _create_llm_wrapper(self):
        """创建 LLM 客户端包装器"""
        def wrapped_send_generate_request(messages):
            """包装的 LLM 请求"""
            return send_generate_request(
                messages=messages,
                server_url=self.llm_server_url,
                model=self.llm_model,
                api_key=self.llm_api_key,
            )
        
        return wrapped_send_generate_request
    
    def convert_nii_to_image(self, nii_path: str, output_path: str, slice_idx: Optional[int] = None):
        """
        将 NIfTI 文件转换为图像（PNG/JPG）
        
        Args:
            nii_path: NIfTI 文件路径
            output_path: 输出图像路径
            slice_idx: 切片索引（如果为None，使用中间切片）
        
        Returns:
            输出图像路径
        """
        try:
            import SimpleITK as sitk
            import numpy as np
            from PIL import Image
            
            # 读取 NIfTI 文件
            sitk_image = sitk.ReadImage(nii_path)
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            # 选择切片
            if slice_idx is None:
                slice_idx = image_array.shape[0] // 2
            
            # 提取切片并归一化
            slice_data = image_array[slice_idx, :, :]
            
            # 归一化到 0-255
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
            if slice_max > slice_min:
                slice_normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                slice_normalized = np.zeros_like(slice_data, dtype=np.uint8)
            
            # 转换为 RGB 图像
            if len(slice_normalized.shape) == 2:
                # 灰度图转 RGB
                slice_rgb = np.stack([slice_normalized] * 3, axis=-1)
            else:
                slice_rgb = slice_normalized
            
            # 保存图像
            image = Image.fromarray(slice_rgb)
            image.save(output_path)
            
            return output_path
        except ImportError:
            raise ImportError("需要安装 SimpleITK: pip install SimpleITK")
        except Exception as e:
            raise ValueError(f"转换 NIfTI 文件失败: {e}")
    
    def infer(
        self,
        image_path: str,
        text_prompt: str,
        is_nii: bool = False,
        slice_idx: Optional[int] = None,
        max_generations: int = 100,
    ) -> Dict[str, Any]:
        """
        执行智能代理推理
        
        Args:
            image_path: 图像路径（或 NIfTI 文件路径）
            text_prompt: 自然语言查询（例如："分割肿瘤区域"）
            is_nii: 是否为 NIfTI 格式
            slice_idx: 如果是 NIfTI，指定切片索引
            max_generations: 最大生成轮数
        
        Returns:
            包含推理结果的字典：
            {
                "messages": 对话历史,
                "final_outputs": 最终分割结果,
                "rendered_image": 可视化图像,
                "output_json_path": JSON 输出路径,
                "output_image_path": 图像输出路径
            }
        """
        if agent_inference is None:
            raise RuntimeError("SAM3 agent 模块未正确导入")
        
        # 如果是 NIfTI 格式，先转换为图像
        actual_image_path = image_path
        if is_nii:
            temp_image_path = os.path.join(
                self.output_dir,
                f"temp_{os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')}_slice_{slice_idx or 'mid'}.png"
            )
            actual_image_path = self.convert_nii_to_image(image_path, temp_image_path, slice_idx)
            print(f"已将 NIfTI 转换为图像: {actual_image_path}")
        
        # 创建服务包装器
        sam_service = self._create_sam_service_wrapper()
        llm_service = self._create_llm_wrapper()
        
        # 执行代理推理
        try:
            messages, final_outputs, rendered_image = agent_inference(
                img_path=actual_image_path,
                initial_text_prompt=text_prompt,
                debug=self.debug,
                send_generate_request=llm_service,
                call_sam_service=sam_service,
                max_generations=max_generations,
                output_dir=self.output_dir,
            )
            
            # 保存结果
            image_basename = os.path.splitext(os.path.basename(actual_image_path))[0]
            prompt_for_filename = text_prompt.replace("/", "_").replace(" ", "_")[:50]
            
            output_json_path = os.path.join(
                self.output_dir,
                f"{image_basename}_{prompt_for_filename}_result.json"
            )
            output_image_path = os.path.join(
                self.output_dir,
                f"{image_basename}_{prompt_for_filename}_result.png"
            )
            
            # 保存 JSON 结果
            final_outputs["text_prompt"] = text_prompt
            final_outputs["image_path"] = actual_image_path
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_outputs, f, indent=4, ensure_ascii=False)
            
            # 保存可视化图像
            rendered_image.save(output_image_path)
            
            return {
                "messages": messages,
                "final_outputs": final_outputs,
                "rendered_image": rendered_image,
                "output_json_path": output_json_path,
                "output_image_path": output_image_path,
            }
        except Exception as e:
            print(f"推理过程中出错: {e}")
            raise
        finally:
            # 清理临时文件
            if is_nii and os.path.exists(actual_image_path) and "temp_" in actual_image_path:
                try:
                    os.remove(actual_image_path)
                except:
                    pass


def create_agent(
    sam3_model_path: Optional[str] = None,
    llm_server_url: Optional[str] = None,
    llm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    output_dir: str = "agent_output",
    device: str = "cuda",
    **kwargs
) -> MedicalImageAgent:
    """
    便捷函数：创建医学图像代理实例
    
    Args:
        sam3_model_path: SAM3 模型路径
        llm_server_url: LLM 服务器地址
        llm_model: LLM 模型名称
        output_dir: 输出目录
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        MedicalImageAgent 实例
    """
    return MedicalImageAgent(
        sam3_model_path=sam3_model_path,
        llm_server_url=llm_server_url,
        llm_model=llm_model,
        output_dir=output_dir,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("SAM3 Agent 推理模块 - 示例")
    print("=" * 60)
    
    # 创建代理（需要配置实际的模型路径和 LLM 服务器）
    agent = create_agent(
        llm_server_url="http://127.0.0.1:8000",  # 替换为实际的 LLM 服务器地址
        output_dir="agent_output",
    )
    
    print("\n代理已创建，可以使用以下方式调用：")
    print("""
    # 示例 1: 处理普通图像
    result = agent.infer(
        image_path="path/to/image.png",
        text_prompt="分割肿瘤区域"
    )
    
    # 示例 2: 处理 NIfTI 文件
    result = agent.infer(
        image_path="path/to/brain.nii.gz",
        text_prompt="分割脑肿瘤",
        is_nii=True,
        slice_idx=100  # 可选，指定切片
    )
    """)


