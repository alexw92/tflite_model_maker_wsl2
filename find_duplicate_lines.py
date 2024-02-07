import csv
from collections import defaultdict

def find_duplicate_lines(input_file):
    duplicate_count = 0
    line_counts = defaultdict(int)

    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Assuming the first row is the header, adjust if not

        for row_number, row in enumerate(reader, start=2):  # Start counting rows from 2
            row_string = ','.join(row)  # Convert the row to a string
            line_counts[row_string] += 1

    for count in line_counts.values():
        if count > 1:
            duplicate_count += 1

    return duplicate_count

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    duplicate_count = find_duplicate_lines(input_file)

    print(f"Number of duplicate lines in '{input_file}': {duplicate_count}")
