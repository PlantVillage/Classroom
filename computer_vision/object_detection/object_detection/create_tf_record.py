# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet 
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import sys

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('images_dir', '/Users/petermccloskey/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/computer_vision/banana/data/training_data/images', 'Root directory to raw pet dataset.')
flags.DEFINE_string('annotations_dir', '/Users/petermccloskey/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/computer_vision/banana/data/training_data/annotations', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/Users/petermccloskey/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('trainval_dir', '/Users/petermccloskey/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/computer_vision/banana/data/training_data/trainvals/', 'Path to trainval text file')
flags.DEFINE_string('label_map_path', '/Users/petermccloskey/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/computer_vision/banana/model/configs_and_labels/banana_label_map.pbtxt','Path to label map proto')
flags.DEFINE_string('version', '3', 'The TFrecord version number')
flags.DEFINE_string('model', 'ssd', 'The model that this record is being made for')
flags.DEFINE_string('crop', 'banana', 'The crop that this record is being made for')
flags.DEFINE_boolean('group', False, 'Set grouping function to on or off using boolean')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
											 creating_train,
                       model,
                       example,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename = example+'.jpg'
  #print(filename)
  img_path = os.path.join(image_subdirectory, filename)
  #print(img_path)
  if not os.path.exists(img_path):
    base_filename,ext = os.path.splitext(filename)
  
    if ext == '.jpg':
      img_path = os.path.join(image_subdirectory, base_filename+'.JPG')
      #print('Path with .JPG exists')
    else:
      img_path = os.path.join(image_subdirectory,base_filename + '.jpg')
      #print('Path with .jpg exists')


  if creating_train == True and model is 'ssd':
    resized_height = 300
    resized_width = 300
  	### RESIZE IMAGE ###
    resized_img_path = image_subdirectory + '/resized_' + str(resized_height) + 'x' + str(resized_width) + '/' + filename
  	# Read in original image and decode from jpg
    image_decoded = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
  	## Resized decoded image
    resized_image = tf.image.resize_images(image_decoded, [resized_height, resized_width])
  	## Cast image to 8 bit int and re-encode image
    encoded_jpg = tf.image.encode_jpeg(tf.cast(resized_image,tf.uint8))
  	# Write resized image to file
    fwrite=tf.write_file(resized_img_path, encoded_jpg)
    sess = tf.Session()
    result=sess.run(fwrite) 
    # Read in resized image
    img_path = resized_img_path
  
  with tf.io.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  key = hashlib.sha256(encoded_jpg).hexdigest()

 

  width = int(data['size']['width'])
  height = int(data['size']['height'])
  
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))
    xmin_ = float(obj['bndbox']['xmin'])
    xmax_ = float(obj['bndbox']['xmax'])
    ymin_ = float(obj['bndbox']['ymin'])
    ymax_ = float(obj['bndbox']['ymax'])

    error = False

    if xmin_ > width:
      xmin_ = width
      error = True
      print('XMIN > width for file', filename)
                
    if xmin_ < 0:
        xmin_ = 0
        #error = True
        print('XMIN < 0 for file', filename)
        
    if xmax_ > width:
        xmax_ = width
        #error = True
        print('XMAX > width for file', filename)
    
    if ymin_ > height:
        ymin_ = height
        #error = True
        print('YMIN > org_height for file', filename)
    
    if ymin_ < 0:
        ymin_ = 0
        #error = True
        print('YMIN < 0 for file', filename)
    
    if ymax_ > height:
        ymax_= height
        #error = True
        print('YMAX > org_height for file', filename)
    
    if xmin_ >= xmax_:
        error = True
        print('xmin >= xmax for file', filename)
        print('xmin = ' + str(xmin_))
        print('xmax = ' + str(xmax_))
        
    if ymin_ >= ymax_:
        error = True
        print('ymin >= ymax for file', filename)

    if error:
      sys.exit()

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    class_name = obj['name']
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,
                     model,
										 creating_train):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  

  for idx, example in enumerate(examples):
    print('On image %d out of %d' % (idx, len(examples)))
    if idx % 100 == 0:
      print('\n\nOn image %d of %d\n\n' % (idx, len(examples)))
    path = os.path.join(annotations_dir, example + '.xml')
    print('Image: %s'%examples[idx])

    if not os.path.exists(path):
      print('\nCould not find '+ os.path.basename(path))
      #logging.warning('Could not find %s, ignoring example.', path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    #print(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, label_map_dict, image_dir, creating_train, model, example)
    writer.write(tf_example.SerializeToString())

  writer.close()

def group_images(examples_list):
  example_codes = [] 
  examples_list_grouped = []
  # Adding unique image codes to directory
  for example in examples_list:
    if example[-3:] not in example_codes:
      example_codes.append(example[-3:])
  # Checking all file names for image codes and grouping them with their associated images
  for code in example_codes:
    groupings = []
    for item in examples_list:
      if item[-3:] == code:
        groupings.append(item)
    examples_list_grouped.append(groupings)
  
  return examples_list_grouped

# TODO: Add test for pet/PASCAL main files.
def main(_):
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  version_num = FLAGS.version
  model_name = FLAGS.model
  crop_name = FLAGS.crop
  train_pct = 0.8
  logging.info('Reading from dataset.')
  image_dir = FLAGS.images_dir
  annotations_dir = FLAGS.annotations_dir
  examples_path = os.path.join(FLAGS.trainval_dir, 'trainval_') + crop_name + '_' + version_num + '.txt'
  examples_list = dataset_util.read_examples_list(examples_path)
  #print(examples_list)
  indices = [ i for i, word in enumerate(examples_list) if word.startswith('spore-50') ]
  result_list = [examples_list[i] for i in indices]   
  #print(result_list)
  for i in enumerate(reversed(indices)):
    del examples_list[i[1]]
  
  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)

  
  if FLAGS.group == True:
    examples_list_grouped = group_images(examples_list)
    count = 0
    num_examples = len(examples_list_grouped)
    num_train = int(0.8 * num_examples)
    random.shuffle(examples_list_grouped)
    # Split performed on grouped list rather than list of individual lists
    train_examples_grouped = examples_list_grouped[:num_train]
    val_examples_grouped = examples_list_grouped[num_train:]
    train_examples = []
    # Unraveling groups to form final image list without groupings
    for train_group in train_examples_grouped:
      for train_example in train_group:
        train_examples.append(train_example)
    val_examples = []
    for val_group in val_examples_grouped:
      for val_example in val_group:
        val_examples.append(val_example)
  else: 
    num_examples = len(examples_list)
    num_train = int(0.8 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    train_examples += result_list
    #print(train_examples)
    #print(val_examples)
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
  train_output_path = os.path.join(FLAGS.output_dir, ('train_'+ crop_name + '_' + version_num + '.record'))
  val_output_path = os.path.join(FLAGS.output_dir, 'eval_'+ crop_name + '_' + version_num + '.record')
  print('\n\ncreating VALIDATION record\n\n')
  create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir, val_examples, model_name, creating_train=False)
  print('\n\ncreating TRAIN record\n\n')
  create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_examples, model_name, creating_train=True)
  print('Complete!')
  
if __name__ == '__main__':
  tf.app.run()
