# tflite_model_maker_wsl2
This shows how to set up tflite model maker on WSL2

## Redo it in new env (works!)
0.) You need python38 for doing this!
Oh and **dont even think of building python3810 from source! It will fuck everything up!**
**better: create conda env with python3.8.10**
You can list your conda envs with the command ```conda env list```
however in the conda env i still got the same error:
```UnicodeEncodeError: 'utf-8' codec can't encode character '\udcfc' in position 5: surrogates not allowed```
*Solution:* This was because by hostname on Windows Saugmühle was translated wrongly to Ubuntu with an invalid character.
Running ```sudo hostname Saugmuehle``` and restarting wsl fixed it!

For GPU Support you need WSL2 and Kernel version higher than ```5.10.16.3-microsoft-standard-WSL2```

Tried with 

```
sudo cat /proc/version
Linux version 5.15.79.1-microsoft-standard-WSL2 
```

and now nvidia-smi works!


1.) Because it was not installed correctly I created a new venv **model_maker_venv2**
and installed tensorflow==2.8.4 and packaging==20.9 inside it

then I cloned the repo and installed it

```
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

2.) It seems to be installed now

```bash
pip list
tflite-model-maker            0.4.2     /home/alex/examples/tensorflow_examples/lite/model_maker/pip_package/src

```
2.1) I got an error importing something in modelmaker so I downgraded numpy to 1.23
Then I continued with [this](https://www.tensorflow.org/lite/models/modify/model_maker/object_detection)

```python
import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
```

```python
spec = model_spec.get('efficientdet_lite0')
```

3.) The following results in an error making be think it could not load the data from the gcs bucket
```python
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')
```

4.) But then running this seems to train correctly (currently on cpu only I think, cuda11 could not be found)
```python
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
```
Btw: Here on a fresh installed ubuntu22 with python 3816 I got the following error 
```UnicodeEncodeError: 'utf-8' codec can't encode character '\udcfc' in position 5: surrogates not allowed```
*Solution:* This was because by hostname on Windows Saugmühle was translated wrongly to Ubuntu with an invalid character.
Running ```sudo hostname Saugmuehle``` and restarting wsl fixed it!

It trained fine for a while but then failed because of 

```
ImportError: You must install pycocotools (`pip install pycocotools`) (see github repo at https://github.com/cocodataset/cocoapi) for efficientdet/coco_metric to work.
```

4.1) so I installed it and also installed the next required module:

```bash
pip install -q pycocotools
pip install -q opencv-python-headless==4.1.2.30
```

In case pycocotools installation throws an error you might need to install these

```bash
sudo apt-get install libpq-dev python3.8-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libjpeg-dev zlib1g-dev
sudo apt-get install build-essential
```

then I run the code again and restarted training

```python
spec = model_spec.get('efficientdet_lite0')
```

It worked!

5.) Then I ran eval

```python
model.evaluate(test_data)
```

6.) worked also and then I exported the model

```python
model.export(export_dir='.')
```

Worked and resulted in:

```json
{'AP': 0.20119594, 'AP50': 0.34621373, 'AP75': 0.21721725, 'APs': -1.0, 'APm': 0.5330238, 'APl': 0.19910036, 'ARmax1': 0.17052028, 'ARmax10': 0.34225824, 'ARmax100': 0.39623663, 'ARs': -1.0, 'ARm': 0.7, 'ARl': 0.39222324, 'AP_/Baked Goods': 0.025994863, 'AP_/Salad': 0.5475503, 'AP_/Cheese': 0.19085687, 'AP_/Seafood': 0.021515297, 'AP_/Tomato': 0.22006239}
```

7.) Then I loaded the just exported tflite model to perform evaluation:

```python
model.evaluate_tflite('model.tflite', test_data)
```

worked as well!

```json
{'AP': 0.1789487, 'AP50': 0.31319344, 'AP75': 0.1972646, 'APs': -1.0, 'APm': 0.559314, 'APl': 0.17661628, 'ARmax1': 0.13116644, 'ARmax10': 0.24777183, 'ARmax100': 0.26256573, 'ARs': -1.0, 'ARm': 0.64166665, 'ARl': 0.25821817, 'AP_/Baked Goods': 0.0, 'AP_/Salad': 0.5157693, 'AP_/Cheese': 0.17967021, 'AP_/Seafood': 0.0006441821, 'AP_/Tomato': 0.19865985}
```

# Train on your GPU in WSL
You need WSL2 and Kernel version higher than ```5.10.16.3-microsoft-standard-WSL2```

## Installing cuda on wsl2
I need tensorflow 2.8 therefor Cuda 11.2 from the Nvidia [page](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&Distribution=WSL-Ubuntu&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=deblocal)

```bash
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

## then install cudnn
Important: **Use the archieve file and extract it**
Use cudnn 8.6
then try it out and it should show this:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
2023-02-07 11:27:39.769247: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:922] could not open file to read NUMA node: /sys/bus/pci/devices/0000:09:00.0/numa_node
Your kernel may have been built without NUMA support.
2023-02-07 11:27:39.774772: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:922] could not open file to read NUMA node: /sys/bus/pci/devices/0000:09:00.0/numa_node
Your kernel may have been built without NUMA support.
2023-02-07 11:27:39.774827: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:922] could not open file to read NUMA node: /sys/bus/pci/devices/0000:09:00.0/numa_node
Your kernel may have been built without NUMA support.
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

current error: when trying to train i get an error  even though the file mentioned can be found in the lib64 dir

```
>>> model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
Epoch 1/50
2023-02-07 11:33:29.041737: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 23970816 exceeds 10% of free system memory.
2023-02-07 11:33:29.049214: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 26873856 exceeds 10% of free system memory.
2023-02-07 11:33:29.054961: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 36499968 exceeds 10% of free system memory.
2023-02-07 11:33:29.055949: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 24443136 exceeds 10% of free system memory.
2023-02-07 11:33:29.066611: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51556935 exceeds 10% of free system memory.
2023-02-07 11:33:32.566107: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8600
Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory
Aborted
```

**Solution: This is WSL related!**
Do this:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```
Be sure that your library is in ```/usr/lib/wsl/lib```, to see it you can run

```bash
ldconfig -p | grep cuda
```

In the end the environment vars in .bashrc should look like this

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/lib/wsl/lib
export CUDA_HOME=/usr/local/cuda-11.2
export PATH=$PATH:$CUDA_HOME/bin
```

Now the training should work! :)

# Tried on windows (legacy)
```bash
(model_maker_venv) (base) PS Z:\IdeaRepos\model_maker_od> pip install -q --use-deprecated=legacy-resolver tflite-model-maker
ERROR: Could not find a version that satisfies the requirement scann==1.2.6 (from tflite-model-maker) (from versions: none)
ERROR: No matching distribution found for scann==1.2.6 (from tflite-model-maker)
WARNING: You are using pip version 22.0.4; however, version 23.0 is available.

You should consider upgrading via the 'Z:\IdeaRepos\model_maker_od\model_maker_venv\Scripts\python.exe -m pip install --upgrade pip' command.
```

--> scann requires linux -> Do this on WSL2


# NEXT STEP (legacy):
Do all of this [model maker repo](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/modify/model_maker/object_detection.ipynb#scrollTo=2vvAObmTqglq)
in *WSL*

```bash
sudo apt -y install libportaudio2
pip install -q --use-deprecated=legacy-resolver tflite-model-maker
pip install -q pycocotools
pip install -q opencv-python-headless==4.1.2.30
pip uninstall -y tensorflow && pip install -q tensorflow==2.8.0
```

Error at installing tflite-model-maker

Tried

```bash
pip install tflite-model-maker-nightly
```

with tensorflow2.8 installed in venv

```bash
python3 -m venv model_maker_venv
source model_maker_venv/bin/activate
```

Two dependency problems
```bash
ERROR: grpcio-status 1.51.1 has requirement protobuf>=4.21.6, but you'll have protobuf 3.20.3 which is incompatible.
ERROR: tensorflowjs 3.18.0 has requirement packaging~=20.9, but you'll have packaging 23.0 which is incompatible.
```

Installed packaging - no error

```bash
pip install packaging==20.9
```

Installed protobuf...

```bash
pip install protobuf==4.21.6
```

with errors:

```bash
ERROR: tflite-support-nightly 0.4.4.dev20221103 has requirement protobuf<4,>=3.18.0, but you'll have protobuf 4.21.6 which is incompatible.
ERROR: tensorflow-metadata 1.12.0 has requirement protobuf<4,>=3.13, but you'll have protobuf 4.21.6 which is incompatible.
```

Updated pip and created new venv with just this

```
$ pip install tflite-model-maker
```

resulted in an error
```python
ERROR: tensorflow 2.11.0 has requirement protobuf<3.20,>=3.9.2, but you'll have protobuf 3.20.3 which is incompatible.
ERROR: scann 1.2.6 has requirement tensorflow~=2.8.0, but you'll have tensorflow 2.11.0 which is incompatible.
ERROR: tensorflowjs 3.18.0 has requirement packaging~=20.9, but you'll have packaging 23.0 which is incompatible.
```

# Another aproach
Permission errors seem to be WSL related because I was in NTFS file system managed by windows
Tried this inside (WSL) ```/home/alex``` inside a venv (model_maker_venv)

```bash
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

```
ERROR: launchpadlib 1.10.13 requires testresources, which is not installed.
ERROR: tensorboard 2.11.2 has requirement protobuf<4,>=3.9.2, but you'll have protobuf 4.21.12 which is incompatible.
ERROR: tensorflow 2.11.0 has requirement protobuf<3.20,>=3.9.2, but you'll have protobuf 4.21.12 which is incompatible.
ERROR: scann 1.2.6 has requirement tensorflow~=2.8.0, but you'll have tensorflow 2.11.0 which is incompatible.
ERROR: tensorflow-metadata 1.12.0 has requirement protobuf<4,>=3.13, but you'll have protobuf 4.21.12 which is incompatible.
ERROR: tensorflowjs 3.18.0 has requirement packaging~=20.9, but you'll have packaging 23.0 which is incompatible.
ERROR: tflite-support 0.4.3 has requirement protobuf<4,>=3.18.0, but you'll have protobuf 4.21.12 which is incompatible.
```

Then did 

```python
pip install tensorflow~=2.8.0 #(this fixed protobuf it seemes)
pip install packaging~=20.9
pip install -e .
Successfully installed tflite-model-maker
```

Then continued model-maker installation with 

```bash
pip install -q pycocotools
pip install -q opencv-python-headless==4.1.2.30
```

Skipped this because I think i did it already by installing tensorflow-2.8 before 

```
pip uninstall -y tensorflow && pip install -q tensorflow==2.8.0 # didnt do this
```

This way seems to install model maker locally so I switched into
the src dir and invoked python3 and tried to import model-maker but I resulted in error
that tensorflowjs was missing so maybe this wasnt installed correctly.

## Redo it in new env (works!)
0.) You need python38 for doing this!
Oh and **dont even think of building python3810 from source! It will fuck everything up!**
**better: create conda env with python3.8.10**
You can list your conda envs with the command ```conda env list```
however in the conda env i still got the same error:
```UnicodeEncodeError: 'utf-8' codec can't encode character '\udcfc' in position 5: surrogates not allowed```
*Solution:* This was because by hostname on Windows Saugmühle was translated wrongly to Ubuntu with an invalid character.
Running ```sudo hostname Saugmuehle``` and restarting wsl fixed it!

For GPU Support you need WSL2 and Kernel version higher than ```5.10.16.3-microsoft-standard-WSL2```

Tried with 

```
sudo cat /proc/version
Linux version 5.15.79.1-microsoft-standard-WSL2 
```

and now nvidia-smi works!

**Update: Maybe this is an error with the encoding of this new wsl**
But echo $LANG is
C.UTF-8 on both ubuntu20 and ubuntu22

1.) Because it was not installed correctly I created a new venv **model_maker_venv2**
and installed tensorflow==2.8.4 and packaging==20.9 inside it

then I cloned the repo and installed it

```
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

2.) It seems to be installed now

```bash
pip list
tflite-model-maker            0.4.2     /home/alex/examples/tensorflow_examples/lite/model_maker/pip_package/src

```
2.1) I got an error importing something in modelmaker so I downgraded numpy to 1.23
Then I continued with [this](https://www.tensorflow.org/lite/models/modify/model_maker/object_detection)

```python
import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
```

```python
spec = model_spec.get('efficientdet_lite0')
```

3.) The following results in an error making be think it could not load the data from the gcs bucket
```python
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')
```

4.) But then running this seems to train correctly (currently on cpu only I think, cuda11 could not be found)
```python
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
```
Btw: Here on a fresh installed ubuntu22 with python 3816 I got the following error 
```UnicodeEncodeError: 'utf-8' codec can't encode character '\udcfc' in position 5: surrogates not allowed```
*Solution:* This was because by hostname on Windows Saugmühle was translated wrongly to Ubuntu with an invalid character.
Running ```sudo hostname Saugmuehle``` and restarting wsl fixed it!

It trained fine for a while but then failed because of 

```
ImportError: You must install pycocotools (`pip install pycocotools`) (see github repo at https://github.com/cocodataset/cocoapi) for efficientdet/coco_metric to work.
```

4.1) so I installed it and also installed the next required module:

```bash
pip install -q pycocotools
pip install -q opencv-python-headless==4.1.2.30
```

In case pycocotools installation throws an error you might need to install these

```bash
sudo apt-get install libpq-dev python3.8-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libjpeg-dev zlib1g-dev
sudo apt-get install build-essential
```

then I run the code again and restarted training

```python
spec = model_spec.get('efficientdet_lite0')
```

It worked!

5.) Then I ran eval

```python
model.evaluate(test_data)
```

6.) worked also and then I exported the model

```python
model.export(export_dir='.')
```

Worked and resulted in:

```json
{'AP': 0.20119594, 'AP50': 0.34621373, 'AP75': 0.21721725, 'APs': -1.0, 'APm': 0.5330238, 'APl': 0.19910036, 'ARmax1': 0.17052028, 'ARmax10': 0.34225824, 'ARmax100': 0.39623663, 'ARs': -1.0, 'ARm': 0.7, 'ARl': 0.39222324, 'AP_/Baked Goods': 0.025994863, 'AP_/Salad': 0.5475503, 'AP_/Cheese': 0.19085687, 'AP_/Seafood': 0.021515297, 'AP_/Tomato': 0.22006239}
```

7.) Then I loaded the just exported tflite model to perform evaluation:

```python
model.evaluate_tflite('model.tflite', test_data)
```

worked as well!

```json
{'AP': 0.1789487, 'AP50': 0.31319344, 'AP75': 0.1972646, 'APs': -1.0, 'APm': 0.559314, 'APl': 0.17661628, 'ARmax1': 0.13116644, 'ARmax10': 0.24777183, 'ARmax100': 0.26256573, 'ARs': -1.0, 'ARm': 0.64166665, 'ARl': 0.25821817, 'AP_/Baked Goods': 0.0, 'AP_/Salad': 0.5157693, 'AP_/Cheese': 0.17967021, 'AP_/Seafood': 0.0006441821, 'AP_/Tomato': 0.19865985}
```



