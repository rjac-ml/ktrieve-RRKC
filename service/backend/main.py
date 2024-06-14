import os
import io
import subprocess
from pathlib import Path
from typing import Optional
from modal import Image, web_endpoint,Mount, Secret, App, asgi_app, enter, exit, gpu, method
#from fastapi.responses import StreamingResponse
from fastapi import Response

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"

def download_model():    
    from huggingface_hub import snapshot_download

    ignore = [
        "*.bin",
        "*.onnx_data",
        "*/diffusion_pytorch_model.safetensors",
    ]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        ignore_patterns=ignore,
    )


app = App("imagen2")

server_image = (
    Image
    .debian_slim(python_version="3.12")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "text-generation==0.7.0",
        "transformers==4.40.0",
        "diffusers==0.27.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
        "huggingface-hub==0.22.2",
        "pydantic==2.6.4",
        "fastapi==0.110.1",
    )
    .run_function(
        download_model, secrets=[Secret.from_name("huggingface-secret")]
    )
)

GPU_CONFIG = gpu.A10G()  # 2 A100s for LLaMA 2 70B

with server_image.imports():
    from transformers import AutoTokenizer
    import torch
    from diffusers import DiffusionPipeline

@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=1,
    container_idle_timeout=60,
    timeout=60 * 10,
    concurrency_limit=1,
    image=server_image,
)
class Server:

    @enter()
    def start_server(self):
        self.base_model = DiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        self.base_model.to("cuda")

        self.refiner = DiffusionPipeline.from_pretrained(
            refiner_model,
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.refiner.to("cuda")


    @method()
    def image_generation(self, query: str, high_noise_frac: Optional[float] = 0.6, n_steps: Optional[int] = 50, webhook: Optional[str] = ""):
        print("="*100)
        print(query)

        latent = self.base_model(
            prompt=query,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        print("latent...")
        
        image = self.refiner(
            prompt=query,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=latent
        ).images[0]
        print("Generation finish...")

        memory_stream = io.BytesIO()
        image.save(memory_stream, format="JPEG")
        memory_stream.seek(0)
        #return Response(memory_stream.getvalue(), media_type="image/jpeg")
        return memory_stream.getvalue()


@app.local_entrypoint()
def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
    image_bytes = Server().image_generation.remote(prompt)
    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)