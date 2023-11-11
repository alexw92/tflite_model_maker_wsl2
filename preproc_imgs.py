from PIL import Image
from tqdm import tqdm
import os
import sys

def resize_images_in_directory(input_directory, output_directory, target_dimension):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of JPEG files in the directory
    jpg_files = [filename for filename in os.listdir(input_directory) if filename.lower().endswith(('.jpg', '.jpeg'))]

    # Iterate over all files in the directory with tqdm for progress visualization
    for filename in tqdm(jpg_files, desc='Resizing images'):
        # Full path to the input image
        input_path = os.path.join(input_directory, filename)

        # Full path to the output image
        output_path = os.path.join(output_directory, filename)

        # Open the image file
        with Image.open(input_path) as img:
            # Calculate the other dimension to maintain the aspect ratio
            width_percent = (target_dimension / float(img.size[0]))
            target_height = int((float(img.size[1]) * float(width_percent)))

            # Resize the image while preserving the aspect ratio
            resized_img = img.resize((target_dimension, target_height), Image.ANTIALIAS)

            # Save the resized image
            resized_img.save(output_path)

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_directory> <output_directory> <target_dimension>")
        sys.exit(1)

    # Get input directory, output directory, and target dimension from command line arguments
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    target_dimension = int(sys.argv[3]) # let's try 800

    # Call the function with the provided arguments
    resize_images_in_directory(input_directory, output_directory, target_dimension)