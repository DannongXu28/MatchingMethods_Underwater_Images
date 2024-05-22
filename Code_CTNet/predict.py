import numpy as np
import torch
from PIL import Image

from ACT import ACT # Import the ACT model
from option import parser # Import the argument parser



if __name__ == "__main__":

    # Parse command line arguments
    args = parser.parse_args()

    # Initialize the ACT model with the parsed arguments
    model = ACT(args)

    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # Move the model to the selected device

    # Load the pre-trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location = device))
    model.eval()

    # Start an infinite loop to take input images and perform detection
    while True:
        # Prompt the user to input the filename of the first image
        image_1 = input('Input image_1 filename:')
        try:
            # Try to open the first image
            image_1 = Image.open(image_1)
        except:
            # If there is an error in opening the image, print an error message and continue the loop
            print('Image_1 Open Error! Try again!')
            continue

        # Prompt the user to input the filename of the second image
        image_2 = input('Input image_2 filename:')
        try:
            # Try to open the second image
            image_2 = Image.open(image_2)
        except:
            # If there is an error in opening the image, print an error message and continue the loop
            print('Image_2 Open Error! Try again!')
            continue

        # Perform detection on the two input images
        probability = model.detect_image(image_1,image_2)

        # Print the similarity probability
        print(probability)

