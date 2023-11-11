from PIL import Image
from tqdm import tqdm
import os

def resize_images_in_directory(directory_path, output_directory, target_dimension):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of JPEG files in the directory
    jpg_files = [filename for filename in os.listdir(directory_path) if filename.lower().endswith(('.jpg', '.jpeg'))]

    # Iterate over all files in the directory with tqdm for progress visualization
    for filename in tqdm(jpg_files, desc='Resizing images'):
        # Full path to the input image
        input_path = os.path.join(directory_path, filename)

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
# Example usage:
input_directory = '/home/alex/allImgs_extracted'
output_directory = '/home/alex/allImgs_extracted_smaller'
target_dimension = 800  # Set your desired dimension (either width or height)

resize_images_in_directory(input_directory, output_directory, target_dimension)
