#!/usr/local/bin/python3

# the Azure ML SDK only works on python 3, and this script uses some python 3.6 features

import os
import azureml.core
from azureml.core import Workspace, Datastore, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import ContainerRegistry, RunConfiguration
from azureml.train.dnn import TensorFlow

ws = Workspace.from_config()

# setup Azure File Storage

ds = Datastore.register_azure_file_share(
  workspace=ws, 
  datastore_name='plantvillage', 
  file_share_name='plantvillage',
  account_name='plantvillage', 
  account_key='oS4shGioAWHV3C8owAMpcPZeispB+VAHq1QP6c2O3utyFtHiG7V/qgR/8I4FLzbpeF4xIuSRYj+rDQW3aORV6Q==',
  overwrite=True)
