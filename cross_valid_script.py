import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import argparse
import gc
import os
import sys
import numpy as np
import json
import argparse

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
parser.add_argument('--only_folds', '-f', nargs='+', type=int, default=[3], help='List of fold numbers to process (0 to 4)')
parser.add_argument('--epochs', '-e', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--batch_size', '-b', type=int, default=20, help='Batch size for training')
parser.add_argument('--model_name', '-m', type=str, default='lite1', choices=['lite0', 'lite1', 'lite2', 'lite3'], help='Model name')

args = parser.parse_args()
# The current code uses Focal loss which has already weighted loss because of alpha and gamma
# Using the arguments in the script
only_folds = args.only_folds
epochs = args.epochs
batch_size = args.batch_size
model_name = 'efficientdet-' + args.model_name

print(f"training with {str(epochs)} epochs, {str(batch_size)} batch_size, folds {only_folds} ")
fold_files = ['4684_cv_fold_0.csv','4684_cv_fold_1.csv','4684_cv_fold_2.csv','4684_cv_fold_3.csv','4684_cv_fold_4.csv']
for fold_i, fold_file in enumerate(fold_files):
    if only_folds is not None and len(only_folds)>0 and fold_i not in only_folds:
        print(f"Skipping fold {fold_i} because not in fold list {only_folds}")
        continue
    custom_model_dir_name = 'model_'+"2488_more_classes_plus_indiv"#str(num_distinct_files)
    model_dir = f"models/{model_name}/{custom_model_dir_name}_e{str(epochs)}_b{str(batch_size)}_cvf_{fold_i}"
    print(f"training for fold number {fold_i} with file {fold_file}")
    #spec = model_spec.get('efficientdet_lite1')
    # check this url to check valid hparam values
    # https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/third_party/efficientdet/hparams_config.py
    spec = object_detector.EfficientDetLite1Spec( # change this also to correct model spec
        model_name = model_name,
     #   model_dir='/home/alex/checkpoints/',
      #  hparams='grad_checkpoint=true,strategy=gpus',
        hparams='strategy=gpus',
        epochs=epochs, batch_size=batch_size,
        steps_per_execution=1, moving_average_decay=0,
        var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
        tflite_max_detections=25
    )
    
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(fold_file)
    
    print(f"train size {len(train_data)} val size {len(validation_data)}")

    model = object_detector.create(train_data, model_spec=spec, train_whole_model=True, validation_data=validation_data)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    label_map = model.model_spec.config.label_map.as_dict()
    # Writing the dictionary to a JSON file
    with open(model_dir+'/label_map.json', 'w') as file:
        json.dump(label_map, file)
        
        
    evaluation_results = model.evaluate(validation_data)
    # Print the evaluation results
    print(f"Evaluation results for fold {fold_i}: {evaluation_results}")
    
    model.export(export_dir=model_dir)
    print(f"exported to model to {model_dir}")
    tf.keras.backend.clear_session()
    del model
    del train_data
    del validation_data
    gc.collect()

print("Cross-Val-Training finished!")