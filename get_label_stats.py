import csv
import argparse

def get_label_stats(input_file):
    label_counts = {}
    label_splits = {'TRAIN': {}, 'TEST': {}, 'VALIDATE': {}}
    all_files = set()

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows[1:]:
        split_name = row[0]
        label_name = row[2]
        data_file_url = row[1]

        all_files.add(data_file_url)

        if label_name not in label_counts:
            label_counts[label_name] = set()

        if split_name not in label_splits:
            label_splits[split_name] = {}

        if label_name not in label_splits[split_name]:
            label_splits[split_name][label_name] = 0

        if data_file_url not in label_counts[label_name]:
            label_counts[label_name].add(data_file_url)
            label_splits[split_name][label_name] += 1

    print("There are %s different files in this set" % len(all_files))
    for label, files in label_counts.items():
        print("%s occurs in %s different files" % (label, len(files)))

    print("\nClass Distribution in Splits:")
    for split, label_counts in label_splits.items():
        total_files = len(all_files)
        print(f"\n{split}:")
        for label, count in label_counts.items():
            percentage = round(count / total_files * 100, 2)
            print(f"{label}: {count} files ({percentage}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Pascal VOC files to MLFlow CSV format')
    parser.add_argument('file', nargs='?', type=str, help='csv file with Google ML format to investigate stats for', default='salads_ml_use.csv')
    args = parser.parse_args()
    get_label_stats(args.file)
