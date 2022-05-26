import torch

class Debug(torch.nn.Module):
    def __init__(self):
        super(Debug, self).__init__()

    def forward(self, x):
        x = torch.add(x, 5)
        x = torch.sub(x, 3)
        x = torch.mul(x, 3)
        return x


net = Debug()
model_name = 'debug.onnx'
dummy_input= torch.randint(low=1, high=3, size=(3,), dtype=int)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
