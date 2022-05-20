from os.path import exists
from keras.models import load_model
import tensorflow as tf
import numpy as np

# face detetor
class HuskyNamer():
    def __init__(self, modelfile="huskynamer.mns.hdf5"):
        modelpath = modelfile if exists(modelfile) else f"./models/{modelfile}"
        self.names = ['Kali', 'Keke', 'Tati', 'Tonti']
        self.input_size = 224, 224
        self.model = load_model(modelpath)
        self.model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])

    def predict(self, tensor):
        tensor = tf.image.resize(tensor, self.input_size, method='nearest')
        tensor = np.expand_dims(tensor, axis=0)
        classes = self.model.predict(np.vstack([tensor]))[0]
        class_id = np.argmax(classes)

        return {"class": self.names[class_id], "class_id": class_id, "score": classes[class_id]}
