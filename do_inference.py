import cv2
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

label_map = {
    1: 'rice',
    2: 'carrot',
    3: 'strawberry',
    4: 'potato',
    5: 'grape',
    6: 'kidney bean',
    7: 'butter',
    8: 'water melon',
    9: 'tofu',
    10: 'lentil',
    11: 'sweet potato',
    12: 'chickpea',
    13: 'cherry',
    14: 'chilli',
    15: 'avocado',
    16: 'raspberry',
    17: 'zucchini',
    18: 'pear',
    19: 'brocoli',
    20: 'tomato',
    21: 'mango',
    22: 'onion',
    23: 'garlic',
    24: 'apple',
    25: 'coucous',
    26: 'quinoa',
    27: 'cucumber',
    28: 'lemon',
    29: 'ananas',
    30: 'plum',
    31: 'cantaloupe',
    32: 'califlower',
    33: 'kiwi',
    34: 'black bean',
    35: 'green bean',
    36: 'bell pepper',
    37: 'banana',
    38: 'spinach',
    39: 'blackberry',
    40: 'blueberry',
    41: 'orange',
    42: 'mushroom',
    43: 'basil',
    44: 'parsley',
    45: 'egg',
    46: 'ginger',
    47: 'lime',
    48: 'pumpkin',
    49: 'cheese'
}
num_classes = 49

# Load the labels into a list
classes = ['???'] * num_classes
for label_id, label_name in label_map.items():
  classes[label_id-1] = label_name
# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

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
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8
  

def main(args):
  TEMP_FILE = args.input_img
  model_path = args.model_url
  DETECTION_THRESHOLD = args.threshold
  output_img = args.output_img
  label_map = {
    1: 'rice',
    2: 'carrot',
    3: 'strawberry',
    4: 'potato',
    5: 'grape',
    6: 'kidney bean',
    7: 'butter',
    8: 'water melon',
    9: 'tofu',
    10: 'lentil',
    11: 'sweet potato',
    12: 'chickpea',
    13: 'cherry',
    14: 'chilli',
    15: 'avocado',
    16: 'raspberry',
    17: 'zucchini',
    18: 'pear',
    19: 'brocoli',
    20: 'tomato',
    21: 'mango',
    22: 'onion',
    23: 'garlic',
    24: 'apple',
    25: 'coucous',
    26: 'quinoa',
    27: 'cucumber',
    28: 'lemon',
    29: 'ananas',
    30: 'plum',
    31: 'cantaloupe',
    32: 'califlower',
    33: 'kiwi',
    34: 'black bean',
    35: 'green bean',
    36: 'bell pepper',
    37: 'banana',
    38: 'spinach',
    39: 'blackberry',
    40: 'blueberry',
    41: 'orange',
    42: 'mushroom',
    43: 'basil',
    44: 'parsley',
    45: 'egg',
    46: 'ginger',
    47: 'lime',
    48: 'pumpkin',
    49: 'cheese'
}
  
  im = Image.open(TEMP_FILE)
  im.thumbnail((512, 512), Image.ANTIALIAS)
  im.save(TEMP_FILE, 'PNG')
  # Load the TFLite model
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  # Run inference and draw detection result on the local copy of the original file
  detection_result_image = run_odt_and_draw_results(
      TEMP_FILE,
      interpreter,
      threshold=DETECTION_THRESHOLD
  )
  # Show the detection result
  img = Image.fromarray(detection_result_image)
  if output_img is '':
    save_url = '/home/alex/result_'+Path(TEMP_FILE).name
  else:
    save_url = output_img  
  img.save(save_url, 'PNG')
  print('prediction saved to '+save_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for object detection tf lite model')
    parser.add_argument('--model_url', type=str, help='The path to your tf-lite model', default='model.tflite')
    parser.add_argument('--threshold', type=int, help='Detection_threshold', default=0.3)
    parser.add_argument('--input_img', type=str, help='Image to feed to the model', default='')
    parser.add_argument('--output_img', type=str, help='Where the output image is stored', default='')
    args = parser.parse_args()
    main(args)
