#!/usr/bin/env python3
"""
测试水印去除器的基本功能
"""

import sys
import os
import numpy as np
import cv2
from decord import VideoReader, VideoWriter

def create_test_mask_video(video_path, output_mask_path, watermark_region=(50, 50, 200, 100)):
    """
    为测试视频创建一个简单的蒙版视频
    watermark_region: (x, y, width, height) 水印区域
    """
    try:
        # 读取原始视频
        vr = VideoReader(video_path)
        frames = []
        
        for i in range(len(vr)):
            frame = vr[i].asnumpy()
            height, width = frame.shape[:2]
            
            # 创建蒙版
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 在指定区域创建白色蒙版（表示水印位置）
            x, y, w, h = watermark_region
            # 确保不超出边界
            x = min(x, width - w)
            y = min(y, height - h)
            mask[y:y+h, x:x+w] = 255
            
            frames.append(mask)
        
        # 保存蒙版视频
        if frames:
            vw = VideoWriter(output_mask_path, fourcc='mp4v', fps=15)
            for frame in frames:
                vw.write(frame)
            vw.release()
            print(f"测试蒙版视频已保存到: {output_mask_path}")
            return True
        else:
            print("无法创建蒙版视频：没有帧数据")
            return False
            
    except Exception as e:
        print(f"创建测试蒙版视频失败: {str(e)}")
        return False

def check_dependencies():
    """检查依赖是否可用"""
    try:
        import torch
        import gradio as gr
        from diffusers.models import AutoencoderKLWan
        print("✓ 所有依赖可用")
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ 依赖缺失: {str(e)}")
        return False

def test_video_loading():
    """测试视频加载功能"""
    print("\n=== 测试视频加载 ===")
    
    # 查找测试视频
    test_video_paths = [
        "./normal_videos/0.mp4",
        "./cartoon/0.mp4"
    ]
    
    for video_path in test_video_paths:
        if os.path.exists(video_path):
            try:
                vr = VideoReader(video_path)
                print(f"✓ 成功加载视频: {video_path}")
                print(f"  - 帧数: {len(vr)}")
                print(f"  - 尺寸: {vr[0].shape}")
                
                # 创建测试蒙版
                mask_path = video_path.replace('.mp4', '_test_mask.mp4')
                if create_test_mask_video(video_path, mask_path):
                    print(f"✓ 测试蒙版创建成功: {mask_path}")
                
                return video_path, mask_path
            except Exception as e:
                print(f"✗ 视频加载失败: {str(e)}")
        else:
            print(f"✗ 视频文件不存在: {video_path}")
    
    return None, None

def main():
    print("=== MiniMax-Remover 水印去除器测试 ===")
    
    # 检查依赖
    if not check_dependencies():
        print("请先安装所需依赖: pip install -r requirements.txt")
        return False
    
    # 测试视频加载
    video_path, mask_path = test_video_loading()
    if not video_path:
        print("✗ 没有找到可用的测试视频")
        return False
    
    print(f"\n=== 测试完成 ===")
    print(f"可以使用以下命令启动水印去除服务:")
    print(f"cd {os.path.dirname(os.path.abspath(__file__))}")
    print(f"python watermark_remover.py")
    print(f"\n测试文件:")
    print(f"- 视频: {video_path}")
    print(f"- 蒙版: {mask_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)