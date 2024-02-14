import torch

ckpt = torch.load("/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/1c9f047f0e1dbe2e1be6f15f5107bf9f74bb425f/native_pytorch_model.pt", mmap=True, weights_only=True, map_location="cuda")

# prints cuda:0
print(ckpt["model"]["layers.31.mlp.w3.weight"].device)

ckpt["model"](torch.randn(10))