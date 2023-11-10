import sys
import os
import pandas as pd
import json

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
output_csv_file = os.path.splitext(input_csv_file)[0] + "_mlflow.csv"

# Check if the output file already exists in the current directory
if os.path.exists(output_csv_file):
    print(f"The output file '{output_csv_file}' already exists. Program terminated.")
    sys.exit(1)

# Read the provided CSV file
df = pd.read_csv(input_csv_file)

# Initialize a list to store the converted annotations
mlflow_annotations = []

# Iterate through each row of the DataFrame
for _, row in df.iterrows():
    image_path = row["image"]
    annotations = json.loads(row["label"])  # Assuming the label column contains JSON-encoded data

    for annotation in annotations:
        label = annotation["rectanglelabels"][0]  # Assuming only one label per annotation
        x = annotation["x"]
        y = annotation["y"]
        width = annotation["width"]
        height = annotation["height"]

        # Calculate x_max and y_max based on x, y, width, and height
        x_max = x + width
        y_max = y + height

        mlflow_annotations.append([image_path, label, x, y, "", "", x_max, y_max, "", ""])

# Create a DataFrame from the converted annotations
mlflow_df = pd.DataFrame(mlflow_annotations, columns=["path", "label", "x_min", "y_min", "", "", "x_max", "y_max", "", ""])

# Save the DataFrame to the output CSV file
mlflow_df.to_csv(output_csv_file, index=False)
