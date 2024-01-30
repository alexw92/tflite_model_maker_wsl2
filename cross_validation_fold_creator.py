import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import argparse
import os

# Setting up command-line argument parsing
parser = argparse.ArgumentParser(description='Perform stratified K-Fold on object detection dataset.')
parser.add_argument('csv_file', type=str, help='Path to the CSV dataset file')
args = parser.parse_args()

# Load the dataset from the provided file path
absolute_path = os.path.abspath(args.csv_file)
dir, file_name = os.path.split(absolute_path)
cross_val_dir = os.path.join(dir, 'cross_val')
df = pd.read_csv(args.csv_file)

# Correcting the column names based on your dataset format
df.columns = ['Split', 'ImagePath', 'Label', 'Other', 'Columns', 'Not', 'Needed', 'For', 'This', 'Calculation', 'Wow']

# Group by ImagePath to ensure all labels for an image stay together
grouped = df.groupby('ImagePath')

# Use the most frequent label in each image for stratification
image_class_counts = grouped['Label'].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index(name='MostCommonLabel')

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = {}
num_imgs = 0
for fold, (train_idx, test_idx) in enumerate(skf.split(image_class_counts['ImagePath'], image_class_counts['MostCommonLabel'])):
    test_image_paths = image_class_counts.iloc[test_idx]['ImagePath'].tolist()
    fold_data = df[df['ImagePath'].isin(test_image_paths)]
    folds[fold] = fold_data
    num_imgs = num_imgs + fold_data['ImagePath'].nunique()
    # Print the distribution of each fold and the number of unique images
    print(f"Fold {fold}:")
    print(f"Number of unique images: {fold_data['ImagePath'].nunique()}")
    print(f"Number of images: {fold_data['ImagePath'].count()}")
    print("Label Distribution:")
    print(fold_data['Label'].value_counts())
    print()

# Creating CSV files and calculating class distributions
class_distributions = {}

for fold, validation_data in folds.items():
    # Combine the other folds to form the training data
    train_data = pd.concat([folds[f] for f in folds if f != fold])

    # Marking the validation and training data
    validation_data['Split'] = 'VALIDATE'
    train_data['Split'] = 'TRAIN'

    # Combine training and validation data
    combined_data = pd.concat([train_data, validation_data])

    # Save to CSV
    filename = f'{num_imgs}_cv_fold_{fold}.csv'
    out_files_folds = os.path.join(cross_val_dir, filename)
    combined_data.to_csv(out_files_folds, index=False, header=False)

    # Collect class distributions for the validation fold
    class_distributions[f'Fold {fold}'] = validation_data['Label'].value_counts()

# Convert class distributions to a DataFrame and save
class_distribution_df = pd.DataFrame(class_distributions)
out_files_distrib = os.path.join(cross_val_dir, f'{num_imgs}_class_distributions.csv')
class_distribution_df.to_csv(out_files_distrib)
