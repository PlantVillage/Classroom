
# Creating TensorFlow Records

Tensorflow Object Detection API reads data using the TFRecord file format. There are a few things we need to prepare before we can generate the TFRecords. 

## Download the Images and Annotations

Download the data from [here](https://www.dropbox.com/s/5oph6gx38s2zw8y/cassava_data.zip?dl=0) and extract it. There should be 200 files (100 images and 100 annotations) per class for 3 classes, totalling 600 files. 

## Create label map

Create a text file called `cassava_label_map.pbtxt` and list all object label names and corresponding label IDs. Use the following format for each object in label_map.pbtxt:  
```bash
item {  
  id: 1  
  name: 'ClassLabel1'  
}  

item {  
  id: 2  
  name: 'ClassLabel2'  
}
```
* Be sure to replace 'ClassLabel2' and 'ClassLabel2' with your actual class 1 and class 2 label names. You can find the label names within the xml files. There are 3 labels corresponding to the 3 classes we will be using. You can assign any id to any name.  

## Create trainval file

Create a text file called `trainval_cassava_1.txt` and list all filenames (without extensions). Your trainval file should look similar to this:
``` bash
CassavaBSD_1  
CassavaBSD_2  
CassavaBSD_3  
CassavaMD_1  
CassavaMD_2  
CassavaMD_3  
```

## Set up the Folder Structure

– data: (contains  folders for annotations, images, trainvals and the cassava_label_map.pbtxt).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– images: (contains the image files in jpg format)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– annotations (contains all the annotation files in xml format)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– trainvals (contains the collection of trainval files)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– tfrecords (contains the generated TF Records)

![alt text](https://www.dropbox.com/s/9rkz9ql1cf782fx/Screen%20Shot%202023-01-24%20at%2012.04.41%20PM.png?dl=0)

## Generate the TFRecord files.

```bash
# From Classroom/computer_vision/object_detection
python object_detection/create_tf_record.py \
    			--images_dir={PATH_TO}/images \
    			--annotations_dir={PATH_TO}/annotations
    			--trainval_dir={PATH_TO}/trainvals
    			--label_map_path={PATH_TO}/cassava_label_map.pbtxt \
    			--output_dir={PATH_TO}/tfrecords \  
                --version=’1’ \  
    			--model=’ssd’ \  
    			--crop=’cassava’ \  
    			--group=False
```

* Note: You need to replace the `{PATH_TO}` variable with the true path to your files and directories. Do not change the values for the `version`, `model`, `crop`, and `group` flags.

If everything was configured correctly, the script will produce two TFRecord files named `cassava_train_1.record` and
`cassava_val_1.record` and save them in the directory that you specified with the `--output_dir` flag.

## Upload Files

Now that you've created the TFRecords, complete the assignment by uploading the 2 TFRecords, label map and trainval file to [this](https://www.dropbox.com/scl/fo/xryhtm1u6eyexvojacc7n/h?dl=0&rlkey=sum646wlpkewzojh2bkl3m1g5) folder. In order to keep your files seperate from others, create a folder with your name on it and upload the 4 files you generated to that folder. 

[https://www.dropbox.com/s/9rkz9ql1cf782fx/Screen%20Shot%202023-01-24%20at%2012.04.41%20PM.png?dl=0]: https://www.dropbox.com/s/9rkz9ql1cf782fx/Screen%20Shot%202023-01-24%20at%2012.04.41%20PM.png?dl=0