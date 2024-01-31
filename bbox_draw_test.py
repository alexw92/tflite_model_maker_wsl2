import imgaug as ia
from imgaug import augmenters as iaa
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import sys

def draw_boxes(image, boxes, output_path):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    absolute_boxes = [(x_min * width, (1-y_max)* height, x_max * width, (1-y_min) * height) for y_min, x_min, y_max, x_max in boxes]
    for box in absolute_boxes:
        draw.rectangle(box, outline="red", width=2)
    image.save(output_path)

def main(seed):
    image_path = '/home/alex/allImgs_extracted_smaller/760b82e8-a0a65ae6be3f55e6_image.jpg'  # Replace with your original image path
    # augmented_image_path = 'path_to_your_augmented_image.jpg'  # Path for the augmented image
    box = [0.569686,0.199451, 1.0, 0.612077]
    # Read and augment the image

    seed=seed
    image = imageio.imread(image_path)
    print(f"dim 0: {image.shape[0]}")
    print(f"dim 1: {image.shape[1]}")
    bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=box[1] * image.shape[1], y1=box[0] * image.shape[0], x2=box[3] * image.shape[1], y2=box[2] * image.shape[0])], shape=image.shape)
    bbs_transformed = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=box[1] * image.shape[1], y1=(1 - box[2]) * image.shape[0],
                   x2=box[3] * image.shape[1], y2=(1 - box[0]) * image.shape[0])
    ], shape=image.shape)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5, seed=seed),  # Horizontal flips
        iaa.Flipud(0.2, seed=seed),  # Vertical flips (if applicable)
        iaa.Multiply((0.8, 1.2), seed=seed),  # Random brightness changes
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scaling
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)} , seed=seed # Translation
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255),  seed=seed) # Adding noise
    ], random_order=True, seed=seed)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs_transformed)

    # Convert bounding boxes to relative format
    boxes = [(bbox.y1 / image.shape[0], bbox.x1 / image.shape[1], bbox.y2 / image.shape[0], bbox.x2 / image.shape[1]) for bbox in bbs.bounding_boxes]
    boxes_aug = [
        (1 - bbox.y2 / image_aug.shape[0], bbox.x1 / image_aug.shape[1],
        1 - bbox.y1 / image_aug.shape[0], bbox.x2 / image_aug.shape[1])
        for bbox in bbs_aug.bounding_boxes
    ]

    # Draw bounding boxes on the original and augmented images
    original_img = Image.open(image_path)
    augmented_img = Image.fromarray(image_aug)

    draw_boxes(original_img, boxes, 'test1.jpg')
    draw_boxes(augmented_img, boxes_aug, 'test2.jpg')

if __name__ == "__main__":
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    else:
        seed = 42    
    main(seed)
