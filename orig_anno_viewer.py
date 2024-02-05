import tkinter as tk
from tkinter import Canvas, Button, Label, ttk
from PIL import Image, ImageTk, ExifTags
import csv
import json
import os
# Simulating reading from a CSV file - for actual use, you would open a file instead
# Path to your CSV file
csv_file_path = './annotations/annotations.csv'
csv_file_path_singles = './annotations/single_annotations.csv'
img_dir = '/home/alex/allImgs_extracted'

# Creating a dictionary to hold the parsed data
parsed_data = {}
labels = set()
rotation_counts = {i: 0 for i in range(1, 9)}
current_rotation_filter = 0 
grocery_filter = None

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

    if rotation == 3:  # 180° rotation
        # Invert both the x and y coordinates
        x, y = 100 - (x + width), 100 - (y + height)
    elif rotation == 6:  # 90° CW rotation
        # Swap x and y, adjust for new origin, swap width and height
        x, y, width, height = y, 100 - (x + width), height, width
    elif rotation == 8:  # 270° CW (90° CCW) rotation
        # Swap x and y, adjust for new origin, swap width and height
        x, y, width, height = 100 - (y + height), x, height, width  
    print(f'Rotation to {rotation}')
    # Return the adjusted or original bounding box
    return {'x': x, 'y': y, 'width': width, 'height': height}


def toggle_coord_rotation():
    pass


def toggle_rotation_filter():
    global current_rotation_filter
    current_rotation_filter = (current_rotation_filter + 1) % 9  # Cycle through 0-8
    rotation_filter_button.config(text=f"Filter: Rotation {current_rotation_filter}")
    update_image_list()
    

def update_image_list():
    # Filter image_paths based on current_rotation_filter
    if current_rotation_filter == 0:
        filtered_paths = list(parsed_data.keys())
    else:
        filtered_paths = [path for path, data in parsed_data.items() if data['rotation'] == current_rotation_filter]
    if grocery_filter is None:
        pass
    else:
        filtered_paths = [
            path
            for path, data in parsed_data.items()
            if any(grocery_filter == label['rectanglelabels'][0] for label in data['label'])
        ]   
    
    # Update global image_paths and reset current_image_index if necessary
    global image_paths, current_image_index
    image_paths = filtered_paths
    if grocery_filter:
        print(f"Selected Grocery: {grocery_filter}")
    print(f"Number of images: {len(image_paths)}")
    current_image_index = 0
    update_image(0) 
    

def get_image_rotation(image_path):
    """Returns the rotation value from image EXIF data, if available."""
    try:
        image = Image.open(image_path)
        exif = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS}
        rotation = exif.get('Orientation', None)
    except (AttributeError, TypeError, KeyError):
        rotation = None
    return rotation

# Opening the CSV file to read data
with open(csv_file_path_singles, mode='r', encoding='utf-8') as file:
    # Using csv.DictReader to read the CSV data from the file
    reader = csv.DictReader(file)

    for row in reader:
        # Extracting the image path to use as a key
        image_path = row['image']
        _, filename = os.path.split(image_path)
        image_path = os.path.join(img_dir, filename)
        
        rotation = get_image_rotation(image_path)
        
        # If rotation data exists, update count
        if rotation in rotation_counts:
            rotation_counts[rotation] += 1

        # Converting the label data from JSON string to a Python dictionary
        label_data = json.loads(row['label'])
        # Storing the data in the dictionary
        parsed_data[image_path] = {
            "annotation_id": row["annotation_id"],
            "annotator": row["annotator"],
            "created_at": row["created_at"],
            "id": row["id"],
            "label": label_data,
            "lead_time": row["lead_time"],
            "updated_at": row["updated_at"],
            "rotation": rotation 
        }
        for l in label_data:
            labels.add(*l['rectanglelabels'])
        

image_paths = list(parsed_data.keys())
current_image_index = 0
for rotation, count in rotation_counts.items():
    print(f"Rotation {rotation}: {count} images")

def draw_image(canvas, image_path, info_label):
    canvas.delete("all")  # Clear the canvas
    image = Image.open(image_path)

    
    width, height = image.size
    scaled_width, scaled_height = 800, int((800 / width) * height) if width > 800 else (width, height)
    image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo  # Keep a reference!
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    # Update info label with file name and rotation info
    info_label.config(text=f"File: {image_path.split('/')[-1]}, Rotation Info: {parsed_data[image_path]['rotation']}")    
    
    # Draw bounding boxes and labels
    for label in parsed_data[image_path]['label']:
        rotatedLabels = adjust_bounding_box_for_rotation(bounding_box=label, rotation=parsed_data[image_path]['rotation'])
        # Convert percentages to pixels for the original dimensions
        original_x = rotatedLabels['x'] * label['original_width'] / 100
        original_y = rotatedLabels['y'] * label['original_height'] / 100
        original_width = rotatedLabels['width'] * label['original_width'] / 100
        original_height = rotatedLabels['height'] * label['original_height'] / 100

        # Calculate scaled coordinates
        scale_x = scaled_width / label['original_width']
        scale_y = scaled_height / label['original_height']
        x1 = original_x * scale_x
        y1 = original_y * scale_y
        x2 = x1 + (original_width * scale_x)
        y2 = y1 + (original_height * scale_y)

        canvas.create_rectangle(x1, y1, x2, y2, outline='red')
        canvas.create_text(x1, y1, text=label['rectanglelabels'][0], fill="yellow", anchor=tk.SW)


def update_image(direction):
    global current_index
    current_index += direction
    current_index = max(0, min(len(image_paths) - 1, current_index))
    draw_image(canvas, image_paths[current_index], info_label)
    
    
def on_selection_change(event):
    global grocery_filter
    # Get the current selection
    selected_option = combo_box.get()
    if selected_option == "":
        grocery_filter = None
    else:
        grocery_filter = selected_option
    update_image_list()

current_index = 0

root = tk.Tk()
root.title("Image Viewer")

canvas = Canvas(root, width=800, height=1000)
canvas.pack()

info_label = Label(root, text="", wraplength=800)
info_label.pack()

rotation_coords_button = Button(root, text="Rotation Coords", command=toggle_coord_rotation)
rotation_coords_button.pack()

rotation_filter_button = Button(root, text="Filter: Rotation 0", command=toggle_rotation_filter)
rotation_filter_button.pack()

# Setup Next and Previous buttons to navigate images
next_button = Button(root, text="Next", command=lambda: update_image(1))
next_button.pack(side="right")

prev_button = Button(root, text="Previous", command=lambda: update_image(-1))
prev_button.pack(side="left")

# Dropdown list setup
labels.add("")
options = list(labels) # Example options
options.sort()
combo_box = ttk.Combobox(root, values=options)
combo_box.pack()
combo_box.bind("<<ComboboxSelected>>", on_selection_change)

draw_image(canvas, image_paths[current_image_index], info_label)

root.mainloop()