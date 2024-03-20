import sys
import os
import pandas as pd
import json
from PIL import Image, ExifTags
from tqdm import tqdm 

def adjust_bounding_box_for_rotation(bounding_box, rotation):
    """
    Adjust bounding boxes for specific image rotations to match rotation mode 1 orientation.

    Args:
    - bounding_box: A dict with keys 'x', 'y', 'width', 'height' (all in percentages of the image's dimensions).
    - rotation: The rotation mode of the image (1, 3, 6, 8).
    - image_width, image_height: Original dimensions of the image.

    Returns:
    - Adjusted bounding box as a dict with 'x', 'y', 'width', 'height'.
    """
    x, y, width, height = [bounding_box[k] for k in ('x', 'y', 'width', 'height')]
    global rotations_from_3, rotations_from_6, rotations_from_8

    if rotation == 3:  # 180° rotation
        # Invert both the x and y coordinates
        x, y = 100 - (x + width), 100 - (y + height)
        rotations_from_3 += 1
    elif rotation == 6:  # 90° CW rotation
        # Swap x and y, adjust for new origin, swap width and height
        rotations_from_6 += 1
        x, y, width, height = y, 100 - (x + width), height, width
    elif rotation == 8:  # 270° CW (90° CCW) rotation
        # Swap x and y, adjust for new origin, swap width and height
        rotations_from_8 += 1
        x, y, width, height = 100 - (y + height), x, height, width  
    # Return the adjusted or original bounding box
    elif rotation == 0 or rotation == 1 or rotation is None:
        pass
    else:
        print(f"ERROR: unknown rotation in image: {rotation}")
    return {'x': x, 'y': y, 'width': width, 'height': height}


def get_image_rotation(image_path):
    """Returns the rotation value from image EXIF data, if available."""
    try:
        image = Image.open(image_path)
        exif = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS}
        rotation = exif.get('Orientation', None)
    except (AttributeError, TypeError, KeyError, FileNotFoundError):
        rotation = None
    return rotation

# Check if the command line argument is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py input_csv_file.csv")
    sys.exit(1)

# Get the input CSV file from the command line argument
input_csv_file = sys.argv[1]

# Check if the file exists and is a CSV file
if not os.path.isfile(input_csv_file) or not input_csv_file.endswith('.csv'):
    print("Error: The provided input file is not a valid CSV file.")
    sys.exit(1)

# Define the output file name
dir_name, file_name = os.path.split(input_csv_file)
file_name_without_ext, ext = os.path.splitext(file_name)
output_csv_file = os.path.join(dir_name, file_name_without_ext + "_mlflow" + ext)
rotations_from_8, rotations_from_6, rotations_from_3 = 0,0,0

# override file if already present
# Check if the output file already exists in the current directory
# if os.path.exists(output_csv_file):
#    print(f"The output file '{output_csv_file}' already exists. Program terminated.")
#    sys.exit(1)

orig_img_dir = '/home/alex/allImgs_extracted'

print(f"Assuming original image dir is {orig_img_dir}")
# Read the provided CSV file
df = pd.read_csv(input_csv_file)

# Initialize a list to store the converted annotations
mlflow_annotations = []

# Iterate through each row of the DataFrame
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_path = row["image"]
    _, image_name = os.path.split(image_path)
    image_path = os.path.join(orig_img_dir, image_name)
    rotation = get_image_rotation(image_path)
    annotations = json.loads(row["label"])  # Assuming the label column contains JSON-encoded data

    for annotation in annotations:
        label = annotation["rectanglelabels"][0] 
        # Original bounding box values (in percentages or pixels)
        bounding_box = {
            'x': annotation["x"],
            'y': annotation["y"],
            'width': annotation["width"],
            'height': annotation["height"]
        }
        
        # Adjust the bounding box for the image rotation
        # THIS IS SUPER IMPORTANT AND THE REASON WE STILL NEED THE ORIG IMAGES
        adjusted_box = adjust_bounding_box_for_rotation(bounding_box, rotation)
        
        # Use the adjusted bounding box values for further processing
        x = adjusted_box['x']
        y = adjusted_box['y']
        width = adjusted_box['width']
        height = adjusted_box['height']

        # Calculate x_max and y_max based on x, y, width, and height
        x_max = x + width
        y_max = y + height

        mlflow_annotations.append([image_path, label, x, y, "", "", x_max, y_max, "", ""])
print(f"#Rotations from 3: {rotations_from_3}\n#Rotations from 6: {rotations_from_6}\nRotations from 8: {rotations_from_8}")
# Create a DataFrame from the converted annotations
mlflow_df = pd.DataFrame(mlflow_annotations, columns=["path", "label", "x_min", "y_min", "", "", "x_max", "y_max", "", ""])

# Save the DataFrame to the output CSV file
mlflow_df.to_csv(output_csv_file, index=False)
