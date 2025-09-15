import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int32Bias,Int8BiasPerTensorFloatInternalScaling
import numpy as np
import os
from brevitas.export import export_onnx_qcdq


def export_qonnx_model(model, filename="lenet_qat.qonnx"):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    #export_qonnx(model, args=dummy_input, export_path=filename)
    BrevitasONNXManager.export(model, dummy_input, export_path=filename)
    print(f"Exported quantized model to {filename}")
    
# ---------------------------
# Brevitas LeNet QAT architecture
# ---------------------------
class BrevitasLeNetQAT(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, weight_bits=8, act_bits=8):
        super().__init__()
        wq = Int8WeightPerTensorFixedPoint
        aq = Int8ActPerTensorFixedPoint
        bq = Int8BiasPerTensorFloatInternalScaling
        self.quant_inp = QuantIdentity(bit_width=act_bits, act_quant=aq,return_quant_tensor=True,name="IdentityLayer")

        # Conv layers with bias_quant=bq
        self.conv1 = QuantConv2d(in_channels, 6, 5, stride=1, padding=0,
                                 weight_bit_width=weight_bits, weight_quant=wq,
                                 bias=True, bias_quant=bq, cache_inference_quant_bias=True, name="conv1")
        
        self.relu1 = QuantReLU(bit_width=act_bits, act_quant=aq,return_quant_tensor=True, name="relu1")
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = QuantConv2d(6, 16, 5, stride=1, padding=0,
                                 weight_bit_width=weight_bits, weight_quant=wq,
                                 bias=True, bias_quant=bq, cache_inference_quant_bias=True, name="conv2")
        
        self.relu2 = QuantReLU(bit_width=act_bits, act_quant=aq,return_quant_tensor=True,name="relu2")
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers with bias_quant=bq
        self.fc1 = QuantLinear(16 * 4 * 4, 120, weight_bit_width=weight_bits, weight_quant=wq,
                               bias=True, bias_quant=bq, cache_inference_quant_bias=True, name="fc1")
        
        self.relu3 = QuantReLU(bit_width=act_bits, act_quant=aq,return_quant_tensor=True, name="relu3")

        self.fc2 = QuantLinear(120, 84, weight_bit_width=weight_bits, weight_quant=wq,
                               bias=True, bias_quant=bq, cache_inference_quant_bias=True, name="fc2")
        self.relu4 = QuantReLU(bit_width=act_bits, act_quant=aq,return_quant_tensor=True, name="relu4")

        self.fc3 = QuantLinear(84, num_classes, weight_bit_width=weight_bits, weight_quant=wq,
                               bias=True, bias_quant=bq, cache_inference_quant_bias=True, name="fc3")

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Training QAT on MNIST
# ---------------------------
def train_qat(model, num_epochs=10, batch_size=128, lr=0.01, device='cpu'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"Starting QAT training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Save QAT model
    torch.save(model.state_dict(), "brevitas_lenet_qat.pth")
    print("Saved QAT model: brevitas_lenet_qat.pth")

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Perform a forward pass to populate the bias cache for export
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        #_ = model(dummy_input)
        quant_input = model.quant_inp(dummy_input)  # quantized input tensor with scale info
        _ = model(quant_input)

    return model, test_loader


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrevitasLeNetQAT()

    # Train and then get the model instance
    trained_model, _ = train_qat(model, num_epochs=200, batch_size=128, lr=0.01, device=device)
    trained_model.eval()
    trained_model = trained_model.to("cpu")
    export_onnx_qcdq(trained_model, torch.randn(1, 1, 28, 28), export_path='lenet_qat.onnx')
