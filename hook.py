import torch
import torch.nn as nn

a = torch.randn(5)
a.to(non_blocking=True)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def move_to_cpu_hook(module, input, output):
    """
    Hook to be called after forward pass to move module's parameters to CPU.
    """
    # Move parameters and buffers to CPU
    for param in module.parameters():
        param.data = param.data.cpu()
    for buffer in module.buffers():
        buffer.data = buffer.data.cpu()
    
    print(module)

# Initialize model and move it to GPU if available
model = MyModel()
if torch.cuda.is_available():
    model.cuda()

# Register the hook for layer1
hook = model.layer1.register_forward_hook(move_to_cpu_hook)

# Example input
input = torch.randn(5, 10)
if torch.cuda.is_available():
    input = input.cuda()

# Forward pass
output = model(input)

# Don't forget to remove the hook if it's no longer needed
hook.remove()
