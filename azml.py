from azureml.core.workspace import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DockerConfiguration

from azureml.core import Experiment, Environment, ScriptRunConfig, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice, LocalWebservice


# interactive login
ws = Workspace(
    workspace_name = "yan-ml-space",
    subscription_id = "e57785da-412d-4cda-92c4-e1b5b3dca10a",
    resource_group = "Cognitive",
)
# ws.get_details()

# http://patorjk.com/software/taag/#p=display&v=0&f=Small

#   _____         _      _           
#  |_   _| _ __ _(_)_ _ (_)_ _  __ _ 
#    | || '_/ _` | | ' \| | ' \/ _` |
#    |_||_| \__,_|_|_||_|_|_||_\__, |
#                              |___/ 

def do_experiment():
    exp = Experiment(workspace=ws, name="HuskyNames-MobileNetV3Small")

    if True:
        # use our own conda environment
        env = Environment("user-managed-env")
        env.python.user_managed_dependencies = True
        src = ScriptRunConfig(source_directory='./', script='huskynames.py', environment=env)

    if False:
        # use Docker, installed on local compute
        env = Environment("local-docker-env")
        env.python.user_managed_dependencies = False
        env.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
        dc = DockerConfiguration(use_docker=True)
        src = ScriptRunConfig(source_directory='./', script='azml_train.py', environment=env, docker_runtime_config=dc)


    run = exp.submit(src)
    # run.wait_for_completion(show_output=True)


#   ___           _                        _      __                 _                 _ 
#  |   \ ___ _ __| |___ _  _ _ __  ___ _ _| |_   / _|_ _ ___ _ __   | |   ___  __ __ _| |
#  | |) / -_) '_ \ / _ \ || | '  \/ -_) ' \  _| |  _| '_/ _ \ '  \  | |__/ _ \/ _/ _` | |
#  |___/\___| .__/_\___/\_, |_|_|_\___|_||_\__| |_| |_| \___/_|_|_| |____\___/\__\__,_|_|
#           |_|         |__/                                                             
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python

model = Model.register(ws, model_name="face", model_path="./models/facehaardetector.xml")
# print(model.name, model.id, model.version, sep='\t')

env = Environment.from_conda_specification(
    name = "tf_cv_mtcnn",
    file_path = "./envs/az_tf_cv_mtcnn.yaml",
)

inference_config = InferenceConfig(
    environment=env,
    source_directory="./serving",
    entry_script="./az_face.py",
)


# local webservice will build a docker image locally
# deployment_config = LocalWebservice.deploy_configuration(port=8002)
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores = 1,
    memory_gb = 4,
)

service = Model.deploy(
    workspace=ws,
    name="faceservice",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

# container registry deleted
# https://docs.microsoft.com/en-us/answers/questions/603277/error-404-acideploymentfailed.html
service.wait_for_deployment(show_output=True)
