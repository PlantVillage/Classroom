#!/usr/bin/python3

# the Azure ML SDK only works on python 3, and this script uses some python 3.6 features

import azureml.core
from azureml.core import Workspace, Datastore, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.container_registry import ContainerRegistry
from azureml.core.runconfig import EnvironmentDefinition 
from azureml.train.dnn import TensorFlow

# Change these as needed:
experiment_name = 'ssdlite_mobilenet_v2_banana'
config_file = '/input/tensorflow/pv_app_development/config_files/ssdlite_mobilenet_v2_banana.config'
tf_version = '1.10'

CLUSTER_NAME = 'higher-memory-lp'

# you shouldn't need to change anything below this line

ws = Workspace.from_config()
print(ws)
pv_ds = Datastore.get(ws, 'plantvillage')

# setup remote compute

try:
    compute_target = ComputeTarget(workspace=ws, name=CLUSTER_NAME)
    print('Found existing cluster, using it.')
except ComputeTargetException:
    print('Cluster not found. Creating a new one.')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_ND6',
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, CLUSTER_NAME, compute_config)

compute_target.wait_for_completion(show_output=True)

input_path = pv_ds.as_mount()

script_params = {
    
    '--alsologtostderr': '', 
    '--input_path': input_path,
    '--pipeline_config_path': config_file,
    '--model_dir': 'outputs/'
    
}

conda_packages = [
    'numpy'
]

pip_packages = [
    'numpy',
    'scikit-image',
    'cython',
    'absl-py',
    #'git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
    'git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
]

custom_docker_image="plantvillage-mls2:latest"

image_registry = ContainerRegistry()
image_registry.address='plantvillage6562d480.azurecr.io'

tf = TensorFlow(
    entry_script='object_detection/model_main.py',
    script_params=script_params,
    source_directory='research',
    compute_target=compute_target,
    conda_packages=conda_packages,
    pip_packages=pip_packages,
    use_gpu=True,
    use_docker=True,
    image_registry_details=image_registry,
    custom_docker_image=custom_docker_image,
    framework_version=tf_version
)

exp = Experiment(workspace=ws, name=experiment_name)

run = exp.submit(tf)
print(run.get_portal_url())
run.wait_for_completion(show_output = True)


