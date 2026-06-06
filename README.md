# AMD DML Shim — Run Axolotl on AMD GPU + Windows

Fine-tune LLMs on an AMD GPU on Windows. No NVIDIA. No Linux. No dual boot.

## What this is

A Python interception layer that makes AI training software believe it has an NVIDIA CUDA GPU, while actually routing compute to your AMD card via Microsoft DirectML.

Built in one session by [@santosthegreat](https://x.com/ArquanimitasDef)

## Hardware tested on

- Windows 10
- AMD Threadripper 2990WX (32 cores / 64 threads / 64GB RAM)
- AMD RX 7700 XT (12GB VRAM)
- Raspberry Pi 5 + Hailo-8 (inference observer over direct ethernet)

## How it works

The shim patches torch internals at Python startup via sitecustomize.py. It blocks torch._C._cuda_init() from crashing, spoofs cuda.is_available() to True, spoofs cuda.get_device_name() to return your AMD card name, and routes all device calls to privateuseone:0 which is DirectML. Training software never knows the difference.

## Setup

Install WSL2 and Ubuntu 22.04, then inside Ubuntu run the following.

Install PyTorch and DirectML:

    pip install torch==2.4.1 torchvision==0.19.1 torch-directml

Copy dml_shim.py to your site-packages and add the import to sitecustomize.py. Then patch torch:

    sed -i 's/torch._C._cuda_init()/pass/' path/to/torch/cuda/__init__.py
    sed -i 's/torch._C._cuda_setDevice(device)/pass/' path/to/torch/cuda/__init__.py
    sed -i 's/if device < 0 or device >= device_count():/if False:/' path/to/torch/cuda/__init__.py

Verify it works:

    python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

Should print True and your AMD card name.

## The Pi Observer

While Windows trains, a Raspberry Pi 5 with Hailo-8 loads checkpoints and runs inference letting you chat with the model in real time as it learns. Connected via direct ethernet at sub-1ms latency. Each piece of hardware does what it was built for.

## Status

- GPU detected via DirectML
- torch.cuda spoofed successfully
- Axolotl loads and starts training
- Mistral 7B LoRA fine-tune running
- Pi inference observer (coming soon)

## License

MIT
