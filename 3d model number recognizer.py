import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tkinter import *

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32*7*7)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model and set to eval mode
model = CNNModel()
model.eval()

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Visualize function
def visualize_layer(layer_output, selected_filter):
    plt.figure(figsize=(10, 10))
    num_filters = layer_output.shape[1]
    
    if selected_filter is None:
        selected_filter = 0  # default to the first filter
    
    # Squeeze the batch dimension if necessary
    filter_output = layer_output[0, selected_filter, :, :].detach().numpy()

    plt.imshow(filter_output, cmap='gray')
    plt.title(f'Filter {selected_filter}')
    plt.show()

# Process and visualize the layer output
def process_and_visualize(current_layer, current_filter):
    images, _ = next(iter(test_loader))  # Load one batch of images

    # Ensure the input tensor has the correct 4D shape [batch_size, channels, height, width]
    if images.dim() == 5:  # Handle 5D inputs
        images = torch.squeeze(images, dim=1)  # Remove extra dimension

    if current_layer == 'conv1':
        conv1_output = model.conv1(images)
        visualize_layer(conv1_output, current_filter)
    elif current_layer == 'conv2':
        conv2_output = model.conv2(images)
        visualize_layer(conv2_output, current_filter)

# Tkinter GUI for selecting layers and filters
class LayerVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Layer Visualizer")
        self.current_layer = 'conv1'
        self.current_filter = 0
        
        self.layer_button_frame = Frame(root)
        self.layer_button_frame.pack(side=TOP)
        
        self.filter_button_frame = Frame(root)
        self.filter_button_frame.pack(side=BOTTOM)
        
        # Layer buttons
        self.conv1_button = Button(self.layer_button_frame, text="Conv Layer 1", command=lambda: self.update_layer('conv1'))
        self.conv1_button.pack(side=LEFT)
        
        self.conv2_button = Button(self.layer_button_frame, text="Conv Layer 2", command=lambda: self.update_layer('conv2'))
        self.conv2_button.pack(side=LEFT)
        
        # Filter buttons
        self.filter_buttons = []
        for i in range(16):  # Assuming 16 filters in the first layer
            btn = Button(self.filter_button_frame, text=f"Filter {i}", command=lambda i=i: self.update_filter(i))
            btn.pack(side=LEFT)
            self.filter_buttons.append(btn)

    def update_layer(self, layer):
        self.current_layer = layer
        process_and_visualize(self.current_layer, self.current_filter)

    def update_filter(self, filter_idx):
        self.current_filter = filter_idx
        process_and_visualize(self.current_layer, self.current_filter)

# Run the Tkinter GUI
root = Tk()
app = LayerVisualizerApp(root)
root.mainloop()
