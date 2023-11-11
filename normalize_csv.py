import csv
import sys

PRECISION = 6

def normalize_coordinates(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for line in reader:
            normalized_line = normalize_line(','.join(line))
            writer.writerow(normalized_line.split(','))

def normalize_line(line):
    fields = line.split(',')

    # Skip lines with insufficient fields
    if len(fields) < 8:
        return line

    # Extract numeric values and normalize
    values = [float(field) if field else None for field in fields[3:9]]
    normalized_values = [round(value / 100, PRECISION) if value is not None else '' for value in values]

    # Replace the original values with normalized values
    fields[3:9] = normalized_values

    # Join the fields back into a CSV line
    normalized_line = ','.join(map(str, fields))

    return normalized_line

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    # Get input path and output path from command line arguments
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Call the function with the provided arguments
    normalize_coordinates(input_path, output_path)