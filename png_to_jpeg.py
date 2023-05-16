import csv
from PIL import Image
import os

def convert_files_to_jpeg(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        converted_rows = []

        for row in reader:
            file_path = row[1]  # Assuming the file path is in the second column

            if os.path.isfile(file_path):
                file_name, file_ext = os.path.splitext(file_path)
                if file_ext.lower() == ".png":
                    jpeg_file_path = file_name + ".jpeg"
                    convert_png_to_jpeg(file_path, jpeg_file_path)
                    print(f"Converted {file_path} to {jpeg_file_path}")
                    row[1] = jpeg_file_path  # Update the file path in the CSV row
                else:
                    print(f"Ignored {file_path}: Not a PNG file")
            else:
                print(f"Ignored {file_path}: File does not exist")

            converted_rows.append(row)

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
