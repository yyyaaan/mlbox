
from keras.models import load_model
import tensorflow as tf
import numpy as np
from os.path import exists

# object detector
class HuskyDetector():
    def __init__(self, modelfile="huskydetector.tflite", threshold = 0.2):
        modelpath = modelfile if exists(modelfile) else f"./models/{modelfile}"
        self.threshold = threshold
        self.names = ['Kali', 'Keke', 'Tati', 'Tonti']
        self.interpreter = tf.lite.Interpreter(model_path=modelpath)
        self.interpreter.allocate_tensors() 
        self.input_size = self.interpreter.get_input_details()[0]['shape'][[2,1]]
        self.input_index = self.interpreter.get_input_details()[0]['index']

        sorted_indices = sorted([x['index'] for x in self.interpreter.get_output_details()])
        self.i_box, self.i_cls, self.i_score = sorted_indices[0:3] # count is not used

    def detect(self, tensor):
        image_height, image_width, _ = tensor.shape
        tensor = tf.image.resize(tensor, self.input_size, method='nearest')
        tensor = np.expand_dims(tensor, axis=0)
        
        # run model
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()

        # assemble output
        boxes = np.squeeze(self.interpreter.get_tensor(self.i_box))
        classes = np.squeeze(self.interpreter.get_tensor(self.i_cls))
        scores = np.squeeze(self.interpreter.get_tensor(self.i_score))

        results = []
        for i, box in enumerate(boxes):
            y_min, x_min, y_max, x_max = box
            results.append({
                "bounding_box": {
                    "top": int(y_min * image_height),
                    "left": int(x_min * image_width),
                    "bottom": int(y_max * image_height),
                    "right": int(x_max * image_width),
                },
                "box": box,
                "class_id": int(classes[i]),
                "class": self.names[int(classes[i])],
                "score": scores[i]
            })
        return [r for r in results if r["score"] > self.threshold]

