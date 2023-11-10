import csv
from PIL import Image
import os
import sys
from tqdm import tqdm
import imghdr

def convert_files_to_jpeg(csv_file):
    converted_files = []
    ignored_files = []
    ignored_not_found_files = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        converted_rows = []

        for row in tqdm(reader):
            file_path = row[1]  # Assuming the file path is in the second column

            if os.path.isfile(file_path):
                file_name, file_ext = os.path.splitext(file_path)
                if file_ext.lower() == ".png":
                    jpeg_file_path = file_name + ".jpeg"
                    if file_name not in converted_files:
                        convert_png_to_jpeg(file_path, jpeg_file_path)
                        converted_files.append(file_name)
                    row[1] = jpeg_file_path  # Update the file path in the CSV row
                elif 'jpeg' != imghdr.what(file_path):
                    file_name, file_ext = os.path.splitext(file_path)
                    convert_png_to_jpeg(file_path, file_path)
                    converted_files.append(file_name)                
                else:
                    ignored_files.append(file_path)
            else:
                ignored_not_found_files.append(file_path)

            converted_rows.append(row)
    print(f"Converted {len(converted_files)}")    
    print(f"Ignored {len(ignored_files)} files because: Not a PNG file")
    print(f"Ignored {len(ignored_not_found_files)} files because: Not found")  

    # Write the updated CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(converted_rows)

def convert_png_to_jpeg(png_file, jpeg_file):
    image = Image.open(png_file)
    image = image.convert("RGB")
    image.save(jpeg_file, "JPEG")

# Usage
if __name__ == '__main__':
    if len(sys.argv) != 2:
        csv_file = "/mnt/z/IdeaRepos/tflite_model_maker_wsl2/merged_annotations_mlflow.csv"
    else:
        csv_file = sys.argv[1]
    convert_files_to_jpeg(csv_file)
    
