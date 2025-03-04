import torch
from torch import nn
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, AutoProcessor, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from flux.transformer_flux import FluxTransformer2DModel

from flux.pipeline_flux_chameleon import FluxPipeline
# from flux.pipeline_flux_img2img import FluxImg2ImgPipeline
# from flux.pipeline_flux_inpaint import FluxInpaintPipeline
# from flux.pipeline_flux_controlnet import FluxControlNetPipeline, FluxControlNetModel
# from flux.pipeline_flux_controlnet_img2img import FluxControlNetImg2ImgPipeline
# from flux.controlnet_flux import FluxMultiControlNetModel
# from flux.pipeline_flux_controlnet_inpainting import FluxControlNetInpaintPipeline

from qwen2_vl.modeling_qwen2_vl import Qwen2VLSimplifiedModel
import os
import cv2
import numpy as np
import math

def get_model_path(model_name):
    """Get the full path for a model based on the checkpoints directory."""
    base_dir = os.getenv('CHECKPOINT_DIR', '/mnt/nvme1/yao/pretrained/Qwen2vl-Flux')  # Allow environment variable override
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

class FluxModel:
    def __init__(self, is_turbo=False, device="cuda", required_features=None):
        """
        Initialize FluxModel with specified features
        Args:
            is_turbo: Enable turbo mode for faster inference
            device: Device to run the model on
            required_features: List of required features ['controlnet', 'depth', 'line', 'sam']
        """
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        if required_features is None:
            required_features = []

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

    def process_image(self, image):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ]
            }
        ]
        text = self.qwen2vl_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        with torch.no_grad():
            inputs = self.qwen2vl_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
            output_hidden_state, image_token_mask, image_grid_thw = self.qwen2vl(**inputs)
            image_hidden_state = output_hidden_state[image_token_mask].view(1, -1, output_hidden_state.size(-1))

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
        if prompt != "":
            self.qwen2vl_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen2vl'], min_pixels=256*28*28, max_pixels=256*28*28)
            t5_prompt_embeds = self.compute_t5_text_embeddings(prompt=prompt, device=self.device)
            t5_prompt_embeds = self.t5_context_embedder(t5_prompt_embeds)
        else:
            self.qwen2vl_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen2vl'], min_pixels=512*28*28, max_pixels=512*28*28)

        qwen2_hidden_state_a, image_grid_thw_a = self.process_image(input_image_a)
        # 只有当所有注意力参数都被提供时，才应用注意力机制
        if mode == "variation":
            if center_x is not None and center_y is not None and radius is not None:
                qwen2_hidden_state_a = self.apply_attention(qwen2_hidden_state_a, image_grid_thw_a, center_x, center_y, radius)
            qwen2_hidden_state_a = self.connector(qwen2_hidden_state_a)

        #############################
        # IMAGE GENERATION
        #############################
        if mode == "variation":
            # Initialize different pipelines
            pipeline = FluxPipeline(
                transformer=self.transformer,
                scheduler=self.noise_scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
            )

            gen_images = pipeline(
                prompt_embeds=qwen2_hidden_state_a.repeat(batch_size, 1, 1),
                t5_prompt_embeds=t5_prompt_embeds.repeat(batch_size, 1, 1) if t5_prompt_embeds is not None else None,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images


        return gen_images
