import csv

def normalize_coordinates(line, precision=6):
    fields = line.split(',')
    
    # Skip lines with insufficient fields
    if len(fields) < 8:
        return line
    
    # Extract numeric values and normalize
    values = [float(field) if field else None for field in fields[3:9]]
    normalized_values = [round(value / 100, precision) if value is not None else '' for value in values]
    
    # Replace the original values with normalized values
    fields[3:9] = normalized_values
    
    # Join the fields back into a CSV line
    normalized_line = ','.join(map(str, fields))
    
    return normalized_line

# Example usage:
input_path = '/mnt/c/Users/Fistus/annotations_mlflow_shuffled.csv'
output_path = '/mnt/c/Users/Fistus/annotations_mlflow_shuffled_n.csv'

with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for line in reader:
        normalized_line = normalize_coordinates(','.join(line))
        writer.writerow(normalized_line.split(','))
