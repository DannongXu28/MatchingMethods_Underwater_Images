import torch
import torch.nn as nn

from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [2, 2, 2, 2, 2] # Sizes of the filters in each layer
        padding = [0, 0, 0, 0, 0] # Padding applied to each layer
        stride = 2 # Stride length for each layer
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()

        # Load VGG16 backbone
        self.vgg = VGG16(pretrained, 3)

        # Remove the average pooling and classifier layers of VGG16
        del self.vgg.avgpool
        del self.vgg.classifier

        # Calculate the flattened shape after VGG16 feature extraction
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])

        # Define the fully connected layers
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   Pass both images through the VGG16 feature extractor
        #------------------------------------------#
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)   
        #-------------------------#
        #   Compute the L1 distance by taking the absolute difference
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   Pass through two fully connected layers
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
