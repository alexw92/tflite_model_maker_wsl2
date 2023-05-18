import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data, validation_data, test_data = object_detector.DataLoader.from_csv('/mnt/z/IdeaRepos/tflite_model_maker_wsl2/merged_annotations_mlflow.csv')

spec = object_detector.EfficientDetLite0Spec(
    model_name='efficientdet-lite0',
    model_dir='/home/alex/checkpoints/',
    hparams='grad_checkpoint=true,strategy=gpus',
    epochs=50, batch_size=8,
    steps_per_execution=1, moving_average_decay=0,
    var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
    tflite_max_detections=25
)

model = object_detector.create(train_data, model_spec=spec, batch_size=8, 
    train_whole_model=True, validation_data=validation_data)
