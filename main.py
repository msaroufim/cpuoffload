import torch

def print_layer_types(model, indent=0):
    for layer in model.children():
        print(' ' * indent + str(type(layer)))
        print_layer_types(layer, indent + 2)

def print_layer_device(model):
    for name, param in model.named_parameters():
        # breakpoint()
        print(name, param.device)

def offload_by_layer_type(model, layer_type, device="cpu"):
    for layer in model.children():
        if isinstance(layer, layer_type):
            layer.to(device)
        offload_by_layer_type(layer, layer_type, device)

def jit_offload(model, device="cpu"):
    return torch.jit.script(model).to(device

def download_model():
    ## import a model from torch.hub
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    return model
    print("Model loaded successfully")

    # print name of each layer

    # print_layer_types(model)


    # offload_by_layer_type(model, torch.nn.Conv2d, "cpu")

    # print_layer_types(model)

if __name__ == "__main__":
    model = download_model()
    print_layer_device(model)
    model.cuda()
    print_layer_device(model)

    print("Done!")


def parameter_offload_cpu(model : torch.nn.Module, device : torch.device):
    for param in model.parameters():
        param.data = param.data.to(device)
        param.grad.data = param.grad.data.to(device)
    return model

def gradient_offload_cpu(model : torch.nn.Module, device : torch.device):
    for param in model.parameters():
        param.grad.data = param.grad.data.to(device)
    return model

def optimizer_offload_cpu(optimizer : torch.optim.Optimizer, device : torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer