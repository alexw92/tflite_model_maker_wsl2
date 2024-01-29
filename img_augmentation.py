import imgaug as ia
from imgaug import augmenters as iaa
import imageio.v2 as imageio
import os
from collections import defaultdict
import csv

def load_and_augment_image(image_path, labels_bboxes_relative, output_dir, image_id):
    # Load image
    image = imageio.imread(image_path)

    # Convert relative bboxes to absolute
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=box[1]*image.shape[1], y1=(1-box[0])*image.shape[0],
                       x2=box[3]*image.shape[1], y2=(1-box[2])*image.shape[0])
        for _, box in labels_bboxes_relative
    ], shape=image.shape)
    labels = [label for label, _ in labels_bboxes_relative]

    # print(image.shape[1])# x
    # print(image.shape[0])# y
    
      # Save original image with bounding boxes
      #image_with_bbs_original = bbs.draw_on_image(image, size=2, color=[255, 0, 0])  # Red color for original
      #original_image_path = os.path.join(output_dir, f'original_{image_id}.jpg')
      #imageio.imwrite(original_image_path, image_with_bbs_original)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flips
        iaa.Flipud(0.2),  # Vertical flips (if applicable)
        iaa.Multiply((0.8, 1.2)),  # Random brightness changes
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scaling
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}  # Translation
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Adding noise
    ], random_order=True)

    # Apply augmentations
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Draw augmented bounding boxes on the image
    image_with_bbs = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 255, 0])

    # Save augmented image with bounding boxes
    aug_image_path = os.path.join(output_dir, f'aug_{image_id}.jpg')
    imageio.imwrite(aug_image_path, image_with_bbs)

    # Convert augmented bboxes back to relative
    # (You can save these to a file or use them as needed)
    bbs_aug_relative = [[bbox.y1 / image_aug.shape[0], bbox.x1 / image_aug.shape[1],
                         bbox.y2 / image_aug.shape[0], bbox.x2 / image_aug.shape[1] ] 
                        for bbox in bbs_aug.bounding_boxes]
    augmented_labels_bboxes = list(zip(labels, bbs_aug_relative))
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
def process_fold(fold_number, input_dir, output_dir):
    input_file = os.path.join(input_dir, f'4102_cv_fold_{fold_number}.csv')
    output_file = os.path.join(output_dir, f'aug_4102_cv_fold_{fold_number}.csv')

    # Dictionary to aggregate bounding boxes for each image
    image_data = defaultdict(list)

    # Read and aggregate data
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            image_path = row[1]
            label = row[2]
            bbox = [float(coord) if coord else '' for coord in row[3:7]]
            image_data[image_path].append((label, bbox))

    # Process each image
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # First, write the original data
        for image_path, labels_and_bboxes in image_data.items():
            for label, bbox in labels_and_bboxes:
                original_row = ['TRAIN', image_path, label, bbox[0], bbox[1], '', '', bbox[2], bbox[3], '', '']
                writer.writerow(original_row)

            # Augment and write the augmented data
            augmented_label_bboxes = load_and_augment_image(image_path, labels_and_bboxes, output_dir, image_path)  # image_id is replaced with image_path

            for label, bbox in augmented_label_bboxes:
                # Generate and write the augmented row
                # (Assuming load_and_augment_image saves augmented images in a predictable way)
                augmented_image_path = os.path.join(output_dir, 'aug_' + os.path.basename(image_path))
                augmented_row = ['TRAIN', augmented_image_path, label, bbox[0], bbox[1], '', '', bbox[2], bbox[3], '', '']
                writer.writerow(augmented_row)