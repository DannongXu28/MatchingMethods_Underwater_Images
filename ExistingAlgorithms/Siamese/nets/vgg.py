import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url



class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # Adaptive average pooling to 7x7
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # Fully connected layer
            nn.ReLU(True), # ReLU activation
            nn.Dropout(), # Dropout for regularization
            nn.Linear(4096, 4096), # Another fully connected layer
            nn.ReLU(True), # ReLU activation
            nn.Dropout(), # Dropout for regularization
            nn.Linear(4096, num_classes), # Output layer
        )
        self._initialize_weights() # Initialize weights


    def forward(self, x):
        x = self.features(x) # Pass through feature extraction layers
        x = self.avgpool(x) # Apply average pooling
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.classifier(x) # Pass through classifier
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # Initialize convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Initialize batch normalization layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # Initialize fully connected layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Configuration for the VGG16 network
# 'M' denotes MaxPooling layer
# Numbers denote the number of filters in the Conv2D layers
# e.g., [64, 64, 'M'] means two Conv2D layers with 64 filters followed by a MaxPooling layer
def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # Add MaxPooling layer
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) # Add Conv2D layer
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] # Add BatchNorm and ReLU
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] # Add ReLU
            in_channels = v # Update input channels
    return nn.Sequential(*layers) # Return a sequential container of layers

# Configuration dictionary for VGG16
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}
def VGG16(pretrained, in_channels, **kwargs):
    # Create the VGG model
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        # Load pretrained weights
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model
