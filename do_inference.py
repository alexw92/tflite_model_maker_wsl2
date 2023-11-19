import cv2
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import os
import csv
from tqdm import tqdm

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

label_map = None
num_classes = None
COLORS = None
classes = None


def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
   # color = [int(c) for c in COLORS[class_id]]
   # cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
   # y = ymin - 15 if ymin - 15 > 15 else ymin + 15
   # label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
   # cv2.putText(original_image_np, label, (xmin, y),
   #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)

    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)

    # Get the size of the text to determine the size of the background rectangle
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Draw a filled black rectangle as the background
    cv2.rectangle(original_image_np, (xmin, y - label_height), (xmin + label_width, y), color, cv2.FILLED)

    # Draw the text on top of the background
    cv2.putText(original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)    

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8
  
def load_labels(label_file):
  # load labels from label file
  import json

  # Loading the dictionary from the JSON file
  with open(label_file, 'r') as file:
      label_map = json.load(file)
  label_map = {int(key): value for key, value in label_map.items()}
  # Printing the loaded dictionary
  return label_map  

def convert_to_png(file_path):
    im = Image.open(file_path)
    im.thumbnail((512, 512), Image.ANTIALIAS)
    png_file_path = file_path.replace('.jpg', '.png')
    im.save(png_file_path, 'PNG')
    return png_file_path  

def main(args):
    input_csv = args.input_csv
    model_path = args.model_url
    detection_threshold = args.threshold
    output_dir = args.output_dir
    label_file = args.label_file
    print(f'Predications will be saved to {output_dir}')
    
    os.makedirs(output_dir, exist_ok=True)
  
  # get class names and order
    global label_map, num_classes, classes
    label_map = load_labels(label_file)
    num_classes = len(label_map)

    # Load the labels into a list
    classes = ['???'] * num_classes
    for label_id, label_name in label_map.items():
      classes[label_id-1] = label_name
    # Define a list of colors for visualization
    global COLORS
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

    test_files = set()

    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            split = row[0]
            file_path = row[1]

            if split == "TEST":
                test_files.add(file_path)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    for file_path in tqdm(test_files):
        if not os.path.isfile(file_path):
            print(f"Ignored {file_path}: File does not exist")
            continue

        file_name, file_ext = os.path.splitext(file_path)

        if file_ext.lower() != ".jpg" and file_ext.lower() != ".jpeg":
            print(f"Ignored {file_path}: Not a JPG file")
            continue

        try:
            png_file_path = convert_to_png(file_path)

            # Run inference and draw detection result on the local copy of the original file
            detection_result_image = run_odt_and_draw_results(
                png_file_path,
                interpreter,
                threshold=detection_threshold
            )

            # Save the prediction image
            save_url = os.path.join(output_dir, f"prediction_{Path(file_path).name}")
            img = Image.fromarray(detection_result_image)
            img.save(save_url, 'PNG')
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for object detection tf lite model')
    parser.add_argument('--model_url', type=str, help='The path to your tf-lite model', default='model.tflite')
    parser.add_argument('--threshold', type=int, help='Detection_threshold', default=0.3)
    parser.add_argument('--input_csv', type=str, help='CSV file containing file paths and splits', default='input.csv')
    parser.add_argument('--output_dir', type=str, help='Output dir to save predictions to', default='/home/alex/predictions')
    parser.add_argument('--label_file', type=str, help='The file where the label map is specified', default='/home/alex/label_map.json')
    args = parser.parse_args()
    main(args)