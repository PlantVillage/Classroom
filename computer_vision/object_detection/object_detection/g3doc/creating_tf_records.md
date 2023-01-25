# Creating TensorFlow Records
Tensorflow Object Detection API reads data using the TFRecord file format. There are a few things we need to prepare before we can generate the TFRecords. 

## Create label map
Create a text file called cassava_label_map.pbtxt and list all object label names and corresponding label IDs. Use the following format for each object in label_map.pbtxt:
item {
  id: 1
  name: 'ClassLabel1'
}

item {
  id: 2
  name: 'ClassLabel2'
}

**Be sure to replace 'ClassLabel2' and 'ClassLabel2' with your actual class 1 and class 2 label names. You can find the label names within the xml files. There are 3 labels corresponding to the 3 classes we will be using. You can assign any id to any name. 


## Create trainval file
Create a text file called trainval_cassava_1.txt and list all filenames (without extensions). Your trainval file should look similar to this: 
CassavaBSD_1
CassavaBSD_2
CassavaBSD_3
…
CassavaMD_1
CassavaMD_2
CassavaMD_3

sample script (`create_tf_record.py`) are provided to convert from an image and annotation dataset to
TFRecords.

## Download the Images and Annotations
Download the data from [here](https://www.dropbox.com/s/5oph6gx38s2zw8y/cassava_data.zip?dl=0) and extract it. There should be 200 files (100 images and 100 annotations) per class for 3 classes, totalling 600 files. 

## Set up Folder Structure
– data: (contains  folders for annotations, images, trainvals and the cassava_label_map.pbtxt .
	– images: (contains the image files in jpg format)
– annotations (contains all the annotation files in xml format) 
– trainvals (contains the collection of trainval files)
– tfrecords (contains the generated TF Records)


## Generate the TFRecord files.

```bash
# From tensorflow/models/research/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

You should end up with two TFRecord files named `pascal_train.record` and
`pascal_val.record` in the `tensorflow/models/research/` directory.

The label map for the PASCAL VOC data set can be found at
`object_detection/data/pascal_label_map.pbtxt`.

