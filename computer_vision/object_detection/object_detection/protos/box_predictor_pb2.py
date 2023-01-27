# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/box_predictor.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import hyperparams_pb2 as object__detection_dot_protos_dot_hyperparams__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+object_detection/protos/box_predictor.proto\x12\x17object_detection.protos\x1a)object_detection/protos/hyperparams.proto\"\x90\x03\n\x0c\x42oxPredictor\x12Y\n\x1b\x63onvolutional_box_predictor\x18\x01 \x01(\x0b\x32\x32.object_detection.protos.ConvolutionalBoxPredictorH\x00\x12P\n\x17mask_rcnn_box_predictor\x18\x02 \x01(\x0b\x32-.object_detection.protos.MaskRCNNBoxPredictorH\x00\x12G\n\x12rfcn_box_predictor\x18\x03 \x01(\x0b\x32).object_detection.protos.RfcnBoxPredictorH\x00\x12s\n)weight_shared_convolutional_box_predictor\x18\x04 \x01(\x0b\x32>.object_detection.protos.WeightSharedConvolutionalBoxPredictorH\x00\x42\x15\n\x13\x62ox_predictor_oneof\"\x90\x03\n\x19\x43onvolutionalBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\x14\n\tmin_depth\x18\x02 \x01(\x05:\x01\x30\x12\x14\n\tmax_depth\x18\x03 \x01(\x05:\x01\x30\x12&\n\x1bnum_layers_before_predictor\x18\x04 \x01(\x05:\x01\x30\x12\x19\n\x0buse_dropout\x18\x05 \x01(\x08:\x04true\x12%\n\x18\x64ropout_keep_probability\x18\x06 \x01(\x02:\x03\x30.8\x12\x16\n\x0bkernel_size\x18\x07 \x01(\x05:\x01\x31\x12\x18\n\rbox_code_size\x18\x08 \x01(\x05:\x01\x34\x12&\n\x17\x61pply_sigmoid_to_scores\x18\t \x01(\x08:\x05\x66\x61lse\x12%\n\x1a\x63lass_prediction_bias_init\x18\n \x01(\x02:\x01\x30\x12\x1c\n\ruse_depthwise\x18\x0b \x01(\x08:\x05\x66\x61lse\"\xe4\x02\n%WeightSharedConvolutionalBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12&\n\x1bnum_layers_before_predictor\x18\x04 \x01(\x05:\x01\x30\x12\x10\n\x05\x64\x65pth\x18\x02 \x01(\x05:\x01\x30\x12\x16\n\x0bkernel_size\x18\x07 \x01(\x05:\x01\x33\x12\x18\n\rbox_code_size\x18\x08 \x01(\x05:\x01\x34\x12%\n\x1a\x63lass_prediction_bias_init\x18\n \x01(\x02:\x01\x30\x12\x1a\n\x0buse_dropout\x18\x0b \x01(\x08:\x05\x66\x61lse\x12%\n\x18\x64ropout_keep_probability\x18\x0c \x01(\x02:\x03\x30.8\x12%\n\x16share_prediction_tower\x18\r \x01(\x08:\x05\x66\x61lse\"\x92\x04\n\x14MaskRCNNBoxPredictor\x12<\n\x0e\x66\x63_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\x1a\n\x0buse_dropout\x18\x02 \x01(\x08:\x05\x66\x61lse\x12%\n\x18\x64ropout_keep_probability\x18\x03 \x01(\x02:\x03\x30.5\x12\x18\n\rbox_code_size\x18\x04 \x01(\x05:\x01\x34\x12>\n\x10\x63onv_hyperparams\x18\x05 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12%\n\x16predict_instance_masks\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\'\n\x1amask_prediction_conv_depth\x18\x07 \x01(\x05:\x03\x32\x35\x36\x12 \n\x11predict_keypoints\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x17\n\x0bmask_height\x18\t \x01(\x05:\x02\x31\x35\x12\x16\n\nmask_width\x18\n \x01(\x05:\x02\x31\x35\x12*\n\x1fmask_prediction_num_conv_layers\x18\x0b \x01(\x05:\x01\x32\x12\'\n\x18masks_are_class_agnostic\x18\x0c \x01(\x08:\x05\x66\x61lse\x12\'\n\x18share_box_across_classes\x18\r \x01(\x08:\x05\x66\x61lse\"\xf9\x01\n\x10RfcnBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\"\n\x17num_spatial_bins_height\x18\x02 \x01(\x05:\x01\x33\x12!\n\x16num_spatial_bins_width\x18\x03 \x01(\x05:\x01\x33\x12\x13\n\x05\x64\x65pth\x18\x04 \x01(\x05:\x04\x31\x30\x32\x34\x12\x18\n\rbox_code_size\x18\x05 \x01(\x05:\x01\x34\x12\x17\n\x0b\x63rop_height\x18\x06 \x01(\x05:\x02\x31\x32\x12\x16\n\ncrop_width\x18\x07 \x01(\x05:\x02\x31\x32')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.box_predictor_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BOXPREDICTOR._serialized_start=116
  _BOXPREDICTOR._serialized_end=516
  _CONVOLUTIONALBOXPREDICTOR._serialized_start=519
  _CONVOLUTIONALBOXPREDICTOR._serialized_end=919
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR._serialized_start=922
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR._serialized_end=1278
  _MASKRCNNBOXPREDICTOR._serialized_start=1281
  _MASKRCNNBOXPREDICTOR._serialized_end=1811
  _RFCNBOXPREDICTOR._serialized_start=1814
  _RFCNBOXPREDICTOR._serialized_end=2063
# @@protoc_insertion_point(module_scope)
