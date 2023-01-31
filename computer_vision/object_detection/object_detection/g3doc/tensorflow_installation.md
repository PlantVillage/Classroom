# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "Classroom/computer_vision/object_detection" directory)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (>=1.9.0)
*   Cython
*   contextlib2
*   cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

``` bash
pip install tensorflow==1.15.5
```

The remaining libraries can be installed using pip:

``` bash
pip install --user Cython==0.29.30
pip install --user contextlib2
pip install --user pillow==9.0.1
pip install --user lxml==4.9.1
pip install --user jupyter==1.0.0
pip install --user matplotlib==3.3.4
pip install --user protobuf==3.19.4
pip install --user pycocotools==2.0
```


## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the Classroom/computer_vision/object_detection directory:


``` bash
# From Classroom/computer_vision/object_detection
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the Classroom/computer_vision/object_detection and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
Classroom/computer_vision/object_detection:


``` bash
# From Classroom/computer_vision/object_detection
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system.

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
# From Classroom/computer_vision/object_detection

python object_detection/builders/model_builder_test.py
```
