import csv
import subprocess

def replace_urls(input_file, output_file, local_dir):
    urls = {}
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows:
        gcs_url = row[1]
        if gcs_url in urls:
            row[1] = urls[gcs_url]
        else:
            local_path = f"{local_dir}/{gcs_url.split('/')[-1]}"
            subprocess.run(["gsutil", "cp", gcs_url, local_path])
            urls[gcs_url] = local_path
            row[1] = local_path

    with open(output_file, 'w', newline='') as ff:
        writer = csv.writer(ff)
        writer.writerows(rows)

input_file = "salads_ml_use.csv"
output_file = "output.csv"
local_dir = "images"
replace_urls(input_file, output_file, local_dir)
