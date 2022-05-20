from azureml.core.workspace import Workspace
from azureml.core import Experiment

# interactive login
ws = Workspace(
    subscription_id = "e57785da-412d-4cda-92c4-e1b5b3dca10a",
    resource_group = "Cognitive",
    workspace_name = "yan-ml-space",
)

exp = Experiment(workspace=ws, name="HuskyNames-MobileNetV3Small")

runids = [x.properties for x in exp.get_runs()]


metrics = [x.get_metrics() for x in exp.get_runs()]
len(metrics)

min(metrics[0]['loss']), min(metrics[1]['loss'])
min(metrics[0]['val_loss']), min(metrics[1]['val_loss'])
max(metrics[0]['val_accuracy']), max(metrics[1]['val_accuracy'])

