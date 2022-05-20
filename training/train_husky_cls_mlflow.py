import os
from keras.models import Model
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense, Dropout, Flatten
from keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input


# AZURE mlflow settings
from azureml.core.workspace import Workspace
import mlflow
ws = Workspace(
    subscription_id = "e57785da-412d-4cda-92c4-e1b5b3dca10a",
    resource_group = "Cognitive",
    workspace_name = "yan-ml-space",
)

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("HuskyNames-MobileNetV3Small")
mlflow.start_run(run_name="mlflow-auto-1")
mlflow.tensorflow.autolog(every_n_iter=1)
# end Azure

train_data_dir = "/Users/pannnyan/Documents/DevGit/transfer-learning/husky_names"
BATCH_SIZE = 4
class_subset = sorted(os.listdir(train_data_dir))

train_generator = ImageDataGenerator(
    rotation_range=90,
    brightness_range=[0.1, 0.7],
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.15,
    preprocessing_function=preprocess_input,  # VGG16's preprocess
)


params = dict(
    directory=train_data_dir,
    target_size=(224, 224),
    class_mode="categorical",
    classes=class_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

traingen = train_generator.flow_from_directory(subset="training", **params)

validgen = train_generator.flow_from_directory(subset="validation", **params)


def create_model(input_shape, n_classes, optimizer="rmsprop", fine_tune=0):
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = MobileNetV3Small(include_top=False, weights="imagenet", input_shape=input_shape)

    # Defines how many layers to freeze during training.
    non_trainable_layers = (
        conv_base.layers if fine_tune == 0 else conv_base.layers[:-fine_tune]
    )
    for layer in non_trainable_layers:
        layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation="relu")(top_model)
    top_model = Dense(1072, activation="relu")(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation="softmax")(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# call backs
callbacks = [
    ModelCheckpoint(filepath="huskynames.mns.hdf5", save_best_only=True, verbose=1),
    #EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min"),
    #LogToAzure(run),
]



# fine tuning mdoels
vgg_model_fine = create_model(
    input_shape=(224, 224, 3),
    n_classes=4,
    optimizer=adam_v2.Adam(learning_rate=0.00001),
    fine_tune=2,
)


vgg_history = vgg_model_fine.fit(
    traingen,
    batch_size=BATCH_SIZE,
    epochs=500,
    validation_data=validgen,
    steps_per_epoch=traingen.samples // BATCH_SIZE,
    validation_steps=validgen.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

mlflow.end_run()