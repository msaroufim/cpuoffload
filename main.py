import torch


def download_model():
    ## import a model from torch.hub
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    print("Model loaded successfully")

    # print name of each layer
    for name, param in model.named_parameters():
        print(name)
    
    return model


if __name__ == "__main__":
    download_model()
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