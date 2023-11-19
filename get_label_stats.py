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
        if len(row)<3:
            print(f" Empty row after processing {len(all_files)} rows")
            print(row)
            break
            # raise Exception(f"Line {row} not correct") 
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
            
    class_list = list(label_counts.keys())  # Convert dictionary keys to a list
    classes_string = ", ".join(class_list)  # Join the keys with commas
    
    print("There are %s different files in this set" % len(all_files))
    print("There are %s different classes in this set" % len(label_counts))
    print("Classes: "+classes_string)
    for label, files in label_counts.items():
        print("%s occurs in %s different files" % (label, len(files)))

    print("\nClass Distribution in Splits:")
    for split, label_counts in label_splits.items():
        total_files = len(all_files)
        print(f"\n{split}:")
        for label, count in label_counts.items():
            percentage = round(count / sum(label_counts.values()) * 100, 2)
            print(f"{label}: {count} files ({percentage}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gets label occurences for dataset')
    parser.add_argument('file', nargs='?', type=str, help='csv file with Google ML format to investigate stats for', default='salads_ml_use.csv')
    args = parser.parse_args()
    get_label_stats(args.file)
