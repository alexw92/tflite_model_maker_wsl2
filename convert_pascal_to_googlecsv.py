import argparse
import csv
import os
import xml.etree.ElementTree as ET

def pascal_voc_to_mlflow_csv(pascal_voc_file, mlflow_csv_file):
    """
    Converts a label list generated by labelImg in Pascal VOC format into MLFlow CSV format used by Google.

    Parameters:
    pascal_voc_file (str): Path to the Pascal VOC file to convert.
    mlflow_csv_file (str): Path to the MLFlow CSV file to create.
    """
    with open(pascal_voc_file, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        with open(mlflow_csv_file, 'a', newline='') as mlflow_csv:
            writer = csv.writer(mlflow_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if os.path.getsize(mlflow_csv_file) == 0:
                writer.writerow(['path', 'label', 'x_min', 'y_min', '', '', 'x_max', 'y_max', '', ''])

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                x_min = round(float(bbox.find('xmin').text) / width, 6)
                y_min = round(float(bbox.find('ymin').text) / height, 6)
                x_max = round(float(bbox.find('xmax').text) / width, 6)
                y_max = round(float(bbox.find('ymax').text) / height, 6)
                writer.writerow([
                    os.path.join(os.path.dirname(pascal_voc_file), root.find('filename').text),
                    obj.find('name').text,
                    x_min,
                    y_min,
                    '',
                    '',
                    x_max,
                    y_max,
                    '',
                    ''
                ])

def main(directory):
    """
    Converts all Pascal VOC files in the given directory to a single MLFlow CSV file.

    Parameters:
    directory (str): Path to the directory containing the Pascal VOC files to convert.
    """
    mlflow_csv_file = os.path.join(directory, 'mlflow_csv.csv')
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            pascal_voc_file = os.path.join(directory, file)
            pascal_voc_to_mlflow_csv(pascal_voc_file, mlflow_csv_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Pascal VOC files to MLFlow CSV format')
    parser.add_argument('directory', type=str, help='Path to the directory containing the Pascal VOC files to convert', default = '.')
    args = parser.parse_args()
    main(args.directory)