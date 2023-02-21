import csv
import argparse

def get_label_stats(input_file):
    label_counts = {} # cheese: 3, tomato: 2   means  cheese was in 3 images, tomato in 2
    label_image = {} # cheese: {file1, file2, file3}, tomate: {file2, file4}
    all_files = set()
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows:
        data_file_url = row[1]
        label_name = row[2]
        all_files.add(data_file_url)
        if label_name in label_image:
            files = label_image[label_name]
            if data_file_url not in files:
                files.add(data_file_url)
                label_image[label_name] = files
                label_counts[label_name] = label_counts[label_name]+1
        else:
           files = set()
           files.add(data_file_url)
           label_image[label_name] = files
           label_counts[label_name] = 1

    print("There are %s different files in this set"%len(all_files))
    for k,v in label_counts.items():
        print("%s occurs in %s different files (%s)"%(k,v, (str (round(v/len(all_files)*100,2))+'%')))       




input_file = "salads_ml_use.csv"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Pascal VOC files to MLFlow CSV format')
    parser.add_argument('file', nargs='?', type=str, help='csv file with Google ML format to investigate stats for', default = input_file)
    args = parser.parse_args()
    get_label_stats(args.file)