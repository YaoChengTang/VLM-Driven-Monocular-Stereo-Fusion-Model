import os
import sys
import cv2
import numpy as np
import math
sys.path.insert(0,'qwen2vl-flux')

import torch
from torch import nn
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, AutoProcessor, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from flux.transformer_flux import FluxTransformer2DModel

from flux.pipeline_flux_chameleon import FluxPipeline
from flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from flux.pipeline_flux_inpaint import FluxInpaintPipeline
from flux.pipeline_flux_controlnet import FluxControlNetPipeline, FluxControlNetModel
from flux.pipeline_flux_controlnet_img2img import FluxControlNetImg2ImgPipeline
from flux.controlnet_flux import FluxMultiControlNetModel
from flux.pipeline_flux_controlnet_inpainting import FluxControlNetInpaintPipeline

from qwen2_vl.modeling_qwen2_vl import Qwen2VLSimplifiedModel





def get_model_path(model_name):
    """Get the full path for a model based on the checkpoints directory."""
    base_dir = os.getenv('CHECKPOINT_DIR', './pretrained/Qwen2vl-Flux')  # Allow environment variable override
    return os.path.join(base_dir, model_name)

# Model paths configuration
MODEL_PATHS = {
    'flux': get_model_path('flux'),
    'qwen2vl': get_model_path('qwen2-vl'),
}


ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "2.4:1": (1536, 640),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
}

class Qwen2Connector(nn.Module):
    def __init__(self, input_dim=3584, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ConfidenceVLMFlux:
    def __init__(self, args, is_turbo=False, device="cuda"):
        """
        Initialize FluxModel with specified features
        Args:
            is_turbo: Enable turbo mode for faster inference
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        self._turbo_imported = False

        # Initialize base models (always required)
        self._init_base_models()

        if is_turbo:
            self._enable_turbo()

    def _init_base_models(self):
        """Initialize the core models that are always needed"""
        # Qwen2VL and connector initialization
        self.qwen2vl = Qwen2VLSimplifiedModel.from_pretrained(
            MODEL_PATHS['qwen2vl'], 
            torch_dtype=self.dtype
        )
        self.qwen2vl.requires_grad_(False).to(self.device)

        self.connector = Qwen2Connector(input_dim=3584, output_dim=4096)
        connector_path = os.path.join(MODEL_PATHS['qwen2vl'], "connector.pt")
        if os.path.exists(connector_path):
            connector_state_dict = torch.load(connector_path, map_location=self.device, weights_only=True)
            connector_state_dict = {k.replace('module.', ''): v for k, v in connector_state_dict.items()}
            self.connector.load_state_dict(connector_state_dict)
        self.connector.to(self.dtype).to(self.device)

        # Text encoders initialization
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATHS['flux'], subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_PATHS['flux'], subfolder="text_encoder")
        self.text_encoder_two = T5EncoderModel.from_pretrained(MODEL_PATHS['flux'], subfolder="text_encoder_2")
        self.tokenizer_two = T5TokenizerFast.from_pretrained(MODEL_PATHS['flux'], subfolder="tokenizer_2")

        self.text_encoder.requires_grad_(False).to(self.dtype).to(self.device)
        self.text_encoder_two.requires_grad_(False).to(self.dtype).to(self.device)

        # T5 context embedder
        self.t5_context_embedder = nn.Linear(4096, 3072)
        t5_embedder_path = os.path.join(MODEL_PATHS['qwen2vl'], "t5_embedder.pt")
        t5_embedder_state_dict = torch.load(t5_embedder_path, map_location=self.device, weights_only=True)
        self.t5_context_embedder.load_state_dict(t5_embedder_state_dict)
        self.t5_context_embedder.to(self.dtype).to(self.device)

        # Basic components
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATHS['flux'], subfolder="scheduler", shift=1)
        self.vae = AutoencoderKL.from_pretrained(MODEL_PATHS['flux'], subfolder="vae")
        self.transformer = FluxTransformer2DModel.from_pretrained(MODEL_PATHS['flux'], subfolder="transformer")

        self.vae.requires_grad_(False).to(self.dtype).to(self.device)
        self.transformer.requires_grad_(False).to(self.dtype).to(self.device)

        self.qwen2vl_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen2vl'], min_pixels=256*28*28, max_pixels=256*28*28)

        self.pipeline = FluxPipeline(
                transformer=self.transformer,
                scheduler=self.noise_scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
            )
        print("-"*10, "Completed initialization", "-"*10)

    def _enable_lora(self):
        pass

    def _enable_turbo(self):
        """Enable turbo mode for faster inference"""
        if not self._turbo_imported:
            from optimum.quanto import freeze, qfloat8, quantize
            self._turbo_imported = True

        quantize(
            self.transformer,
            weights=qfloat8,
            exclude=[
                "*.norm", "*.norm1", "*.norm2", "*.norm2_context",
                "proj_out", "x_embedder", "norm_out", "context_embedder",
            ],
        )
        freeze(self.transformer) 

    def recover_2d_shape(self, image_hidden_state, grid_thw):
        batch_size, num_tokens, hidden_dim = image_hidden_state.shape
        _, h, w = grid_thw
        h_out = h // 2
        w_out = w // 2
        # 重塑为 (batch_size, height, width, hidden_dim)
        reshaped = image_hidden_state.view(batch_size, h_out, w_out, hidden_dim)
        return reshaped

    def generate_attention_matrix(self, center_x, center_y, radius, image_shape):
        height, width = image_shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = center_y * height, center_x * width
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        attention = np.clip(1 - distances / (radius * min(height, width)), 0, 1)
        return attention

    def apply_attention(self, image_hidden_state, image_grid_thw, center_x, center_y, radius):
        qwen2_2d_image_embedding = self.recover_2d_shape(image_hidden_state, tuple(image_grid_thw.tolist()[0]))
        attention_matrix = self.generate_attention_matrix(
            center_x, center_y, radius,
            (qwen2_2d_image_embedding.size(1), qwen2_2d_image_embedding.size(2))
        )
        attention_tensor = torch.from_numpy(attention_matrix).to(self.dtype).unsqueeze(0).unsqueeze(-1)
        qwen2_2d_image_embedding = qwen2_2d_image_embedding * attention_tensor.to(self.device)
        return qwen2_2d_image_embedding.view(1, -1, qwen2_2d_image_embedding.size(3))

    def compute_text_embeddings(self, prompt):
        with torch.no_grad():
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(self.device)
            prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=False)
            pooled_prompt_embeds = prompt_embeds.pooler_output
        return pooled_prompt_embeds.to(self.dtype)

    def compute_t5_text_embeddings(
        self,
        max_sequence_length=256,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_two(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_two(text_input_ids.to(device))[0]

        dtype = self.text_encoder_two.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def process_image(self, images):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Are there any transparent or reflective objects? Like mirror, glass, window, showcase and so on? If true, reply to me the list of corner coordinates of each objects in the format of (x1,y1,x2,y2,x3,y3,x4,y4) in the image. If false, reply an empty list of corners."},
                    ]
                }
            ] \
            for image in images
        ]
        texts = [self.qwen2vl_processor.apply_chat_template(message, tokenize=False,\
                                                            add_generation_prompt=True) \
                                                        for message in messages]
        # print("-"*30, f"qwen2vl_processor.apply_chat_template: images {len(images)}, texts {len(texts)}, text {texts[0]}")

        with torch.no_grad():
            inputs = self.qwen2vl_processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
            # print("-"*30, f"qwen2vl_processor: inputs {[(key, val.shape) for key, val in inputs.items()]}")
            output_hidden_state, image_token_mask, image_grid_thw = self.qwen2vl(**inputs)
            # print("-"*30, f"qwen2vl: output_hidden_state {output_hidden_state.shape}, image_token_mask {image_token_mask.shape}, image_grid_thw {image_grid_thw.shape}")
            image_hidden_state = output_hidden_state[image_token_mask].view(len(texts), -1, output_hidden_state.size(-1))

        return image_hidden_state, image_grid_thw



    def generate(self, input_image_a, input_image_b=None, prompt="", guidance_scale=3.5, num_inference_steps=28,
                 aspect_ratio="1:1", center_x=None, center_y=None, radius=None, mode="variation",
                 denoise_strength=0.8, mask_image=None, imageCount=2,
                 line_mode=True, depth_mode=True, line_strength=0.4, depth_strength=0.2):

        batch_size = imageCount
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio. Choose from {list(ASPECT_RATIOS.keys())}")

        width, height = ASPECT_RATIOS[aspect_ratio]

        pooled_prompt_embeds = self.compute_text_embeddings(prompt="")
        t5_prompt_embeds = None
        if len(prompt) != 0:
            t5_prompt_embeds = self.compute_t5_text_embeddings(prompt=prompt, device=self.device)
            t5_prompt_embeds = self.t5_context_embedder(t5_prompt_embeds)
            # print("-"*30, f"t5_prompt_embeds: {t5_prompt_embeds.shape}")
        # else:
        #     self.qwen2vl_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen2vl'], min_pixels=512*28*28, max_pixels=512*28*28)

        # print("-"*30, f"{len(input_image_a)} {input_image_a[0].size} images are inputed into process_image")
        qwen2_hidden_state_a, image_grid_thw_a = self.process_image(input_image_a)
        # print("-"*30, qwen2_hidden_state_a.shape, image_grid_thw_a.shape) 
        # 只有当所有注意力参数都被提供时，才应用注意力机制
        if mode == "variation":
            if center_x is not None and center_y is not None and radius is not None:
                qwen2_hidden_state_a = self.apply_attention(qwen2_hidden_state_a, image_grid_thw_a, center_x, center_y, radius)
            qwen2_hidden_state_a = self.connector(qwen2_hidden_state_a)

        # print("!"*30, f"Before pipeline qwen2_hidden_state_a {qwen2_hidden_state_a.shape}, t5_prompt_embeds {t5_prompt_embeds.shape}, pooled_prompt_embeds {pooled_prompt_embeds.shape}")
        gen_images = self.pipeline(
            prompt_embeds=qwen2_hidden_state_a,
            t5_prompt_embeds=t5_prompt_embeds if t5_prompt_embeds is not None else None,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="tensor",
            show_progress_bar=False,
        )

        gen_images


        return gen_images



from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from accelerate import Accelerator
import re

class PrecisionLoRAVLMFlux(ConfidenceVLMFlux):
    def __init__(self, 
                 lora_rank=8,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 train_stages=["stage1", "stage3"],  # 控制训练哪些阶段的层
                 device="cuda"):
        super().__init__(is_turbo=False, device=device)
        
        # 初始化可训练组件
        self._setup_trainable_components()
        
        # LoRA配置
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self._get_lora_targets(train_stages),
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["connector"]  # 保持connector可训练
        )
        
        # 应用LoRA适配
        self.transformer = get_peft_model(self.transformer, self.lora_config)
        
        # 参数冻结管理
        self._apply_parameter_constraints()
        
        # 训练组件初始化
        self.accelerator = Accelerator()
        self.optimizer = AdamW(self._get_trainable_params(), lr=2e-5)
        print(f"总可训练参数：{sum(p.numel() for p in self._get_trainable_params())/1e9:.2f}B")

    def _setup_trainable_components(self):
        """配置基础可训练组件"""
        # Qwen2VL最后一层解冻
        for name, param in self.qwen2vl.named_parameters():
            if "final_layer" in name:  # 根据实际层名调整
                param.requires_grad = True
                
        # Connector解冻
        self.connector.requires_grad_(True)

    def _get_lora_targets(self, stages):
        """动态生成LoRA目标模块"""
        stage_patterns = {
            "stage1": r"transformer\.stage1\.layers\.\d+\.(attn1\.to_[qv])",
            "stage2": r"transformer\.stage2\.layers\.\d+\.(attn2\.to_[qv])",
            "stage3": r"transformer\.stage3\.layers\.\d+\.(ff\.net\.0\.proj)"
        }
        
        targets = []
        for stage in stages:
            if pattern := stage_patterns.get(stage):
                # 通过正则匹配目标模块
                for name, _ in self.transformer.named_parameters():
                    if re.match(pattern, name):
                        targets.append(name)
        return list(set(targets))  # 去重

    def _apply_parameter_constraints(self):
        """应用参数冻结策略"""
        # 冻结文本相关编码器
        components = [
            self.text_encoder,
            self.text_encoder_two,
            self.t5_context_embedder,
            self.vae,
            self.qwen2vl  # 除最后一层外已冻结
        ]
        
        for component in components:
            for param in component.parameters():
                param.requires_grad = False

    def _get_trainable_params(self):
        """获取所有可训练参数"""
        return [
            {"params": self.connector.parameters()},
            {"params": self.transformer.parameters()},
            {"params": [p for p in self.qwen2vl.parameters() if p.requires_grad]}
        ]

    def prepare_training(self):
        """准备分布式训练环境"""
        (self.transformer,
         self.connector,
         self.optimizer) = self.accelerator.prepare(
             self.transformer, self.connector, self.optimizer
         )

    def training_step(self, batch):
        """训练步骤实现"""
        images, prompts = batch
        
        with self.accelerator.autocast():
            # 图像特征提取
            img_features, _ = self.process_image(images)
            projected_features = self.connector(img_features)
            
            # 文本嵌入
            text_embeds = self.compute_t5_text_embeddings(prompts)
            
            # 前向传播
            outputs = self.transformer(
                image_embeds=projected_features,
                t5_prompt_embeds=text_embeds,
                pooled_prompt_embeds=self.compute_text_embeddings("")
            )
            
            # 重建损失（示例）
            loss = torch.nn.functional.l1_loss(outputs, images)

        # 梯度更新
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def save_adapters(self, output_dir):
        """保存适配器权重"""
        # 保存LoRA权重
        self.transformer.save_pretrained(f"{output_dir}/transformer_lora")
        
        # 保存其他可训练组件
        torch.save({
            "connector": self.connector.state_dict(),
            "qwen2vl_final": [p for p in self.qwen2vl.parameters() if p.requires_grad]
        }, f"{output_dir}/additional_weights.pth")

    def load_adapters(self, input_dir):
        """加载适配器权重"""
        # 加载LoRA
        self.transformer = PeftModel.from_pretrained(
            self.transformer, 
            f"{input_dir}/transformer_lora"
        )
        
        # 加载其他组件
        weights = torch.load(f"{input_dir}/additional_weights.pth")
        self.connector.load_state_dict(weights["connector"])
        qwen2vl_params = [p for p in self.qwen2vl.parameters() if p.requires_grad]
        for p, loaded in zip(qwen2vl_params, weights["qwen2vl_final"]):
            p.data.copy_(loaded)