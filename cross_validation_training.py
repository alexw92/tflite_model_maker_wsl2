import os
import pandas as pd
from tflite_model_maker import object_detector
from sklearn.model_selection import StratifiedGroupKFold

file_path = '/home/alex/tflite_model_maker_wsl2/annotations1400_mlflow_shuffled_n.csv'

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")

# Load your CSV data into DataFrame (assuming file names are in column 0)
data = pd.read_csv(file_path, header=None)

# Extract image paths, labels, and groups from the dataset
image_paths = data.iloc[:, 0]  # Assuming file names are in column 0
labels = data.iloc[:, 1]  # Assuming labels are in column 1
groups = data.iloc[:, 0]  # Assuming file names are in column 0 for grouping

# Perform 5-fold cross-validation with stratification and grouping
num_folds = 5
sgkf = StratifiedGroupKFold(n_splits=num_folds)

# Print distributions of labels in each fold
fold_idx = 0
for train_index, val_index in sgkf.split(image_paths, labels, groups):
    train_fold, val_fold = data.iloc[train_index], data.iloc[val_index]

    train_fold_labels = labels.iloc[train_index]
    val_fold_labels = labels.iloc[val_index]

    train_distribution = train_fold_labels.value_counts(normalize=True)
    val_distribution = val_fold_labels.value_counts(normalize=True)

    print(f"Fold {fold_idx + 1} - Train distribution:\n{train_distribution}")
    print(f"Fold {fold_idx + 1} - Validation distribution:\n{val_distribution}")

    # Convert train and validation data to DataLoader format
    train_data_fold = object_detector.DataLoader.from_pandas(train_fold)
    val_data_fold = object_detector.DataLoader.from_pandas(val_fold)

    # Create and train the model for this fold
    model = object_detector.create(
        train_data_fold,
        model_spec=spec,
        batch_size=8,
        train_whole_model=True,
        validation_data=val_data_fold,
        epochs=50
    )

    # Save the trained model for this fold
    model.save(f"fold_{fold_idx}_model")  # Example: Saving models for each fold

    fold_idx += 1
