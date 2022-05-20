from azureml.core.workspace import Workspace
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DockerConfiguration


# interactive login
ws = Workspace(
    subscription_id = "e57785da-412d-4cda-92c4-e1b5b3dca10a",
    resource_group = "Cognitive",
    workspace_name = "yan-ml-space",
)

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


