import imgaug as ia
from imgaug import augmenters as iaa
import imageio.v2 as imageio
import os
from collections import defaultdict
import csv
from tqdm import tqdm
import argparse

# USE ENV conda activate augment_env

# Important: Global dict do keep track of already augmented files
augmented_images = {}

def load_and_augment_image(image_path, labels_bboxes_relative, output_dir, image_name, augment_seed):
    # Load image
    image = imageio.imread(image_path)

    # Convert relative bboxes to absolute and transform coord system
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=box[0] * image.shape[1], y1=box[1] * image.shape[0],
                   x2=box[2] * image.shape[1], y2=box[3] * image.shape[0])
        for _, _, box in labels_bboxes_relative
    ], shape=image.shape)
    split_labels = [(split, label) for split, label, _ in labels_bboxes_relative]

    # print(image.shape[1])# x
    # print(image.shape[0])# y
    
      # Save original image with bounding boxes
      #image_with_bbs_original = bbs.draw_on_image(image, size=2, color=[255, 0, 0])  # Red color for original
      #original_image_path = os.path.join(output_dir, f'original_{image_id}.jpg')
      #imageio.imwrite(original_image_path, image_with_bbs_original)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5, seed=augment_seed),  # Horizontal flips
        iaa.Flipud(0.2, seed=augment_seed),  # Vertical flips (if applicable)
        iaa.Multiply((0.8, 1.2), seed=augment_seed),  # Random brightness changes
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scaling
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translation
            seed=augment_seed),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), seed=augment_seed)  # Adding noise
    ], random_order=True, seed=augment_seed)

    # Apply augmentations
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Draw augmented bounding boxes on the image
    # image_with_bbs = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 255, 0])
    
    aug_image_path = os.path.join(output_dir, f'aug_{image_name}')
    if os.path.isfile(aug_image_path):
        # just overwrite old augmentations, it only takes one minute to augment all so no problem
        imageio.imwrite(aug_image_path, image_aug)
        # this is not an error if it happens after the first file has been processed
        # print(f"Error: File {aug_image_path} already exists! Duplicate processing will lead to incorrect results!")
    else:
        imageio.imwrite(aug_image_path, image_aug)


    # Convert augmented bboxes back to relative, ensure x1 and y1 are non-negative, and cap x2 and y2 at upper bounds
    bbs_aug_relative = [
        [f"{max(bbox.x1 / image_aug.shape[1], 0):.4f}",  # Ensure x1 is not negative
        f"{max(bbox.y1 / image_aug.shape[0], 0):.4f}",  # Ensure y1 is not negative
        f"{min(bbox.x2 / image_aug.shape[1], 1):.4f}",  # Ensure x2 does not exceed 1
        f"{min(bbox.y2 / image_aug.shape[0], 1):.4f}"]  # Ensure y2 does not exceed 1
        for bbox in bbs_aug.bounding_boxes]


    augmented_labels_bboxes = [(split, label, bbox) for ((split, label), bbox) in zip(split_labels, bbs_aug_relative)]
    return augmented_labels_bboxes
# /0ad959eb-20231222_131714.jpg,Garlic,,,,0.645328,0.6871,,
# TRAIN,/home/alex/allImgs_extracted_smaller/0ad959eb-20231222_131714.jpg,Garlic,0.397682,0.442818,,,0.612393,0.566331,,
# TRAIN,/home/alex/allImgs_extracted_smaller/0ad959eb-20231222_131714.jpg,Garlic,0.380604,0.344922,,,0.606304,0.451052,,

# Example usage
#image_paths = ['/home/alex/allImgs_extracted_smaller/0ad959eb-20231222_131714.jpg', '/home/alex/allImgs_extracted_smaller/69ab98f7-20231222_133334.jpg']  # List of your image paths
#bboxes_relative_all = [[[0.403782, 0.556267, 0.645328, 0.6871], [0.397682, 0.442818, 0.612393, 0.566331],[0.380604, 0.344922, 0.606304, 0.451052]],[[0.318391,0.451052,0.609937,0.68893]]]  # Corresponding bboxes for each image
#output_dir = '/home/alex/augmented'

#for i, (image_path, bboxes_relative) in enumerate(zip(image_paths, bboxes_relative_all)):
#    load_and_augment_image(image_path, bboxes_relative, output_dir, i)
def process_fold(fold_number, input_dir, output_dir, cvprefix, seed):
    input_file = os.path.join(input_dir, f'{cvprefix}_cv_fold_{fold_number}.csv')
    output_file = os.path.join(output_dir, f'aug_{cvprefix}_cv_fold_{fold_number}.csv')
    img_out_dir = '/home/alex/allImgs_extracted_smaller_aug'

    # Dictionary to aggregate bounding boxes for each image
    image_data = defaultdict(list)
    eval_data = defaultdict(list)

    # Read and aggregate data
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            split = row[0]
            image_path = row[1]
            label = row[2]
            bbox = [float(coord) for coord in row[3:9] if coord ]
            if split == 'TRAIN':
                image_data[image_path].append((split, label, bbox))
            if split == 'VALIDATE':
                eval_data[image_path].append((split, label, bbox))

    # Process each image
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # First, write the original data
        for image_path, labels_and_bboxes in tqdm(image_data.items(), desc=f"Processing Fold {fold_number}"):
            for split, label, bbox in labels_and_bboxes:
                original_row = [split, image_path, label, bbox[0], bbox[1], '', '', bbox[2], bbox[3], '', '']
                writer.writerow(original_row)

            if image_path not in augmented_images:
                # Image not augmented before, perform augmentation
                augmented_label_bboxes = load_and_augment_image(image_path, labels_and_bboxes, img_out_dir, os.path.basename(image_path), seed)
                augmented_images[image_path] = augmented_label_bboxes
            else:
                # Use previously augmented data
                augmented_label_bboxes = augmented_images[image_path]          

            for split, label, bbox in augmented_label_bboxes:
                # Generate and write the augmented row

                augmented_image_path = os.path.join(img_out_dir, 'aug_' + os.path.basename(image_path))
                augmented_row = [split, augmented_image_path, label, bbox[0], bbox[1], '', '', bbox[2], bbox[3], '', '']
                writer.writerow(augmented_row)
            
        for file, data in eval_data.items():
            for split, label, bbox in data:
                eval_row = [split, file, label, bbox[0], bbox[1], '', '', bbox[2], bbox[3], '', '']
                writer.writerow(eval_row)
            
                
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Script for creating augmentations for cross-validation folds.')
    
    # Add arguments with default values
    parser.add_argument('--input_dir', type=str, default='annotations/cross_val', help='Input directory for annotations.')
    parser.add_argument('--output_dir', type=str, default='annotations/cross_val', help='Output directory for augmented annotations.')
    parser.add_argument('--cv_prefix', type=str, default='4904', help='Prefix for cross-validation identifier (Dataset image number).')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--seed', type=int, default=42, help='Fixed seed for augmentation (default 42).')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process each fold
    for fold in range(args.folds):
        process_fold(fold, args.input_dir, args.output_dir, args.cv_prefix, args.seed)
