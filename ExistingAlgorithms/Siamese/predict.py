import numpy as np
from PIL import Image

from siamese import Siamese

if __name__ == "__main__":
    # Initialize the Siamese model
    model = Siamese()

    # Infinite loop to allow multiple comparisons
    while True:
        # Get the file path for the first image from user input
        image_1_path = input('Input image_1 filename:')
        try:
            # Try to open the first image
            image_1 = Image.open(image_1_path)
        except:
            # Handle the case where the first image cannot be opened
            print('Image_1 Open Error! Try again!')
            continue

        # Get the file path for the second image from user input
        image_2_path = input('Input image_2 filename:')
        try:
            # Try to open the second image
            image_2 = Image.open(image_2_path)
        except:
            # Handle the case where the second image cannot be opened
            print('Image_2 Open Error! Try again!')
            continue

        # Detect similarity between the two images
        probability = model.detect_image(image_1, image_2)

        # Convert the PyTorch tensor to a Python scalar
        similarity = probability.detach().cpu().numpy().item()

        # Define a similarity threshold. Adjust this based on your specific model and use case
        similarity_threshold = 0.5

        # Determine if images are similar based on the threshold
        if similarity > similarity_threshold:
            print(f"Similarity: {similarity}, Images are Similar")
        else:
            print(f"Similarity: {similarity}, Images are Not Similar")
