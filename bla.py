import torch
import torch.nn as nn
import pickle
import os

def select_layers(model, by_type=None, by_name=None):
    selected_layers = []
    for name, module in model.named_modules():
        if by_type and isinstance(module, by_type):
            selected_layers.append((name, module))
        elif by_name and name in by_name:
            selected_layers.append((name, module))
    return selected_layers

def offload_to_cpu(layer):
    for param in layer.parameters():
        param.data = param.data.cpu()
        if param.grad is not None:
            param.grad = param.grad.cpu()

def save_to_disk(object, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f)

def offload_layer_to_disk(layer, base_path, layer_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    layer_path = os.path.join(base_path, f'{layer_name}.pkl')
    save_to_disk(layer.state_dict(), layer_path)

def offload_optimizer_to_disk(optimizer, base_path, name='optimizer_state.pkl'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    optimizer_path = os.path.join(base_path, name)
    save_to_disk(optimizer.state_dict(), optimizer_path)


model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
base_path = './model_offloads'  # Base directory for saving to disk

# Example: Offload certain layers to CPU and disk
for name, layer in select_layers(model, by_type=nn.Linear):
    offload_to_cpu(layer)
    offload_layer_to_disk(layer, base_path, name)

# Offload optimizer state to disk
offload_optimizer_to_disk(optimizer, base_path)

