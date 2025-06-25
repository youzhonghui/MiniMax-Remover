import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
import time
import random
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline
from diffusers.utils import export_to_video
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip
from huggingface_hub import snapshot_download

# 创建必要的目录
os.makedirs("./model/", exist_ok=True)

def download_remover():
    """下载MiniMax-Remover模型"""
    if not os.path.exists("./model/vae"):
        snapshot_download(repo_id="zibojia/minimax-remover", local_dir="./model/")
        print("Download minimax remover completed")
    else:
        print("Model already exists")

# 下载模型
download_remover()

# 全局配置
random_seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pipeline():
    """初始化MiniMax-Remover管道"""
    vae = AutoencoderKLWan.from_pretrained("./model/vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained("./model/transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained("./model/scheduler")

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)
    return pipe

def load_video_frames(video_path, max_frames=201):
    """加载视频帧"""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = min(len(vr), max_frames)
        frames = [vr[i].asnumpy() for i in range(num_frames)]
        del vr
        return frames, num_frames
    except Exception as e:
        raise gr.Error(f"视频加载失败: {str(e)}")

def load_mask_frames(mask_path, target_frames):
    """加载蒙版帧"""
    try:
        vr = VideoReader(mask_path, ctx=cpu(0))
        if len(vr) < target_frames:
            raise gr.Error(f"蒙版视频帧数({len(vr)})少于目标视频帧数({target_frames})")
        
        masks = []
        for i in range(target_frames):
            mask_frame = vr[i].asnumpy()
            # 转换为灰度蒙版
            if len(mask_frame.shape) == 3:
                mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2GRAY)
            # 二值化蒙版
            mask_frame = (mask_frame > 127).astype(np.float32)
            masks.append(mask_frame)
        del vr
        return masks
    except Exception as e:
        raise gr.Error(f"蒙版加载失败: {str(e)}")

def validate_dimensions(video_frames, mask_frames):
    """验证视频和蒙版尺寸"""
    if len(video_frames) != len(mask_frames):
        raise gr.Error(f"视频帧数({len(video_frames)})与蒙版帧数({len(mask_frames)})不匹配")
    
    video_shape = video_frames[0].shape[:2]
    mask_shape = mask_frames[0].shape[:2]
    
    if video_shape != mask_shape:
        raise gr.Error(f"视频尺寸{video_shape}与蒙版尺寸{mask_shape}不匹配")
    
    return True

def preprocess_for_removal(images, masks):
    """预处理图像和蒙版用于去除"""
    out_images = []
    out_masks = []
    
    for img, msk in zip(images, masks):
        # 根据长宽比调整尺寸
        if img.shape[0] > img.shape[1]:
            img_resized = cv2.resize(img, (480, 832), interpolation=cv2.INTER_LINEAR)
            msk_resized = cv2.resize(msk, (480, 832), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = cv2.resize(img, (832, 480), interpolation=cv2.INTER_LINEAR)
            msk_resized = cv2.resize(msk, (832, 480), interpolation=cv2.INTER_NEAREST)
        
        # 图像归一化到[-1, 1]
        img_resized = img_resized.astype(np.float32) / 127.5 - 1.0
        out_images.append(img_resized)
        
        # 蒙版归一化到[0, 1]
        msk_resized = msk_resized.astype(np.float32)
        msk_resized = (msk_resized > 0.5).astype(np.float32)
        out_masks.append(msk_resized)
    
    arr_images = np.stack(out_images)
    arr_masks = np.stack(out_masks)
    
    return torch.from_numpy(arr_images).half().to(device), torch.from_numpy(arr_masks).half().to(device)

def remove_watermark(video_path, mask_path, max_frames, dilation_iterations, num_inference_steps, progress=gr.Progress()):
    """去除水印主函数"""
    if video_path is None:
        raise gr.Error("请上传视频文件")
    if mask_path is None:
        raise gr.Error("请上传蒙版文件")
    
    progress(0.1, "加载视频...")
    # 加载视频帧
    video_frames, num_frames = load_video_frames(video_path, max_frames)
    
    progress(0.2, "加载蒙版...")
    # 加载蒙版帧
    mask_frames = load_mask_frames(mask_path, num_frames)
    
    progress(0.3, "验证尺寸...")
    # 验证尺寸
    validate_dimensions(video_frames, mask_frames)
    
    progress(0.4, "预处理数据...")
    # 预处理
    img_tensor, mask_tensor = preprocess_for_removal(video_frames, mask_frames)
    mask_tensor = mask_tensor[:,:,:,None]  # 添加通道维度
    
    # 确定输出尺寸
    if mask_tensor.shape[1] < mask_tensor.shape[2]:
        height, width = 480, 832
    else:
        height, width = 832, 480
    
    progress(0.5, "开始去除水印...")
    # 执行推理
    with torch.no_grad():
        out = pipe(
            images=img_tensor,
            masks=mask_tensor,
            num_frames=mask_tensor.shape[0],
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            generator=torch.Generator(device=device).manual_seed(random_seed),
            iterations=int(dilation_iterations)
        ).frames[0]
    
    progress(0.8, "生成视频...")
    # 转换输出格式
    out = np.uint8(out * 255)
    output_frames = [img for img in out]
    
    # 保存视频
    video_file = f"/tmp/{time.time()}-{random.random()}-watermark_removed.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(video_file, codec='libx264', audio=False, verbose=False, logger=None)
    
    progress(1.0, "完成!")
    return video_file

def create_preview(video_path, mask_path):
    """创建预览图，显示第一帧和蒙版叠加效果"""
    if video_path is None or mask_path is None:
        return None
    
    try:
        # 加载第一帧
        vr_video = VideoReader(video_path, ctx=cpu(0))
        first_frame = vr_video[0].asnumpy()
        del vr_video
        
        # 加载第一个蒙版
        vr_mask = VideoReader(mask_path, ctx=cpu(0))
        first_mask = vr_mask[0].asnumpy()
        del vr_mask
        
        # 转换蒙版为灰度
        if len(first_mask.shape) == 3:
            first_mask = cv2.cvtColor(first_mask, cv2.COLOR_RGB2GRAY)
        
        # 调整蒙版尺寸匹配视频
        if first_frame.shape[:2] != first_mask.shape[:2]:
            first_mask = cv2.resize(first_mask, (first_frame.shape[1], first_frame.shape[0]))
        
        # 创建叠加效果
        mask_normalized = (first_mask > 127).astype(np.float32)
        mask_colored = np.stack([mask_normalized * 255, np.zeros_like(mask_normalized), np.zeros_like(mask_normalized)], axis=-1)
        
        # 叠加蒙版到图像上
        overlay = first_frame.astype(np.float32) * 0.7 + mask_colored * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay)
    except Exception as e:
        gr.Warning(f"预览生成失败: {str(e)}")
        return None

# 初始化管道
pipe = get_pipeline()

# HTML标题
title_html = """
<div style='text-align:center; font-size:32px; font-family: Arial, Helvetica, sans-serif; margin-bottom: 20px;'>
  <span style="color:#2196f3;"><b>MiniMax</b></span><span style="color:#f06292;"><b>-Remover</b></span> 水印去除工具
</div>
<div style='text-align:center; font-size:16px; color: #666; margin-bottom: 30px;'>
  上传视频和对应的蒙版文件，自动去除固定位置的水印
</div>
"""

# 创建Gradio界面
with gr.Blocks(title="MiniMax-Remover 水印去除工具") as demo:
    gr.HTML(title_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文件上传")
            video_input = gr.Video(label="上传视频文件", height=300)
            mask_input = gr.Video(label="上传蒙版文件 (与视频同尺寸)", height=300)
            
            preview_btn = gr.Button("🔍 预览叠加效果", variant="secondary")
            preview_output = gr.Image(label="预览 (红色区域为水印位置)", height=200)
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 参数设置")
            max_frames_slider = gr.Slider(
                minimum=10, maximum=201, value=81, step=1,
                label="最大处理帧数", 
                info="限制处理的视频帧数，减少处理时间"
            )
            dilation_slider = gr.Slider(
                minimum=1, maximum=20, value=6, step=1,
                label="蒙版膨胀迭代次数", 
                info="增加此值可扩大去除区域"
            )
            inference_steps_slider = gr.Slider(
                minimum=1, maximum=50, value=6, step=1,
                label="推理步数", 
                info="增加步数可提高质量但会增加处理时间"
            )
            
            remove_btn = gr.Button("🚀 开始去除水印", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            output_video = gr.Video(label="去水印结果", height=400)
    
    with gr.Row():
        gr.Markdown("""
        ### 📋 使用说明
        1. **上传视频文件**: 选择需要去除水印的视频
        2. **上传蒙版文件**: 上传与视频同尺寸的蒙版视频，白色区域表示水印位置
        3. **预览效果**: 点击预览按钮查看蒙版叠加效果
        4. **调整参数**: 根据需要调整处理参数
        5. **开始处理**: 点击"开始去除水印"按钮
        
        ### 💡 提示
        - 蒙版文件必须与视频文件具有相同的尺寸和帧数
        - 蒙版中白色/亮色区域表示需要去除的水印位置
        - 处理时间取决于视频长度和参数设置
        - 建议先用较少帧数测试效果
        """)
    
    # 示例文件
    gr.Examples(
        examples=[
            ["./normal_videos/0.mp4"],
            ["./normal_videos/1.mp4"],
            ["./cartoon/0.mp4"],
            ["./cartoon/1.mp4"],
        ],
        inputs=[video_input],
        label="示例视频"
    )
    
    # 事件绑定
    preview_btn.click(
        create_preview,
        inputs=[video_input, mask_input],
        outputs=preview_output
    )
    
    remove_btn.click(
        remove_watermark,
        inputs=[video_input, mask_input, max_frames_slider, dilation_slider, inference_steps_slider],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001, share=False)