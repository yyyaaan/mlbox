import json
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from base64 import b64encode, b64decode

class FaceAnonymizer:
    def __init__(self, threshold = 0.2):
        self.face_cascade = cv2.CascadeClassifier('face.xml')
    
    def detect_face_tensor(self, tensor, blur=0):
        # Convert into grayscale
        gray = cv2.cvtColor(tensor, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        print(faces)
        if blur > 0:
            for (x, y, w, h) in faces:
                face_area = tensor[y:(y+h), x:(x+w)]
                # blur size must be odd number
                blur_size = (2*int(w/blur/2) + 1, 2*int(h/blur/2) + 1)
                tensor[y:(y+h), x:(x+w)] = cv2.GaussianBlur(face_area, blur_size, 0)
        
        return tensor, faces


# object detector
class HuskyLiteDetector():
    def __init__(self, threshold = 0.2):
        import tensorflow as tf
        self.threshold = threshold
        self.names = ['Kali', 'Keke', 'Tati', 'Tonti']
        self.interpreter = tf.lite.Interpreter(model_path="husky.tflite")
        self.interpreter.allocate_tensors() 
        self.input_size = self.interpreter.get_input_details()[0]['shape'][[2,1]]
        self.input_index = self.interpreter.get_input_details()[0]['index']


        sorted_indices = sorted([x['index'] for x in self.interpreter.get_output_details()])
        self.i_box, self.i_cls, self.i_score = sorted_indices[0:3] # count is not used

    def predict_tensor(self, tensor):
        image_height, image_width, _ = tensor.shape
        tensor = cv2.resize(tensor, self.input_size)
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
                "class_id": int(classes[i]),
                "class": self.names[int(classes[i])],
                "score": scores[i]
            })
        return [r for r in results if r["score"] > self.threshold]

    def predict(self, image_path):
        tensor = cv2.imread(image_path)
        return self.predict_tensor(tensor)



# classifier
class HuskyNamer():
    
    def __init__(self):
        from keras.models import load_model
        self.names = ['Kali', 'Keke', 'Tati', 'Tonti']
        self.input_size = 224, 224
        self.model = load_model("huskynames.mns.hdf5")
        self.model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
        
    def predict_tensor(self, tensor):
        tensor = cv2.resize(tensor, self.input_size)
        tensor = np.expand_dims(tensor, axis=0)
        classes = self.model.predict(np.vstack([tensor]))[0]
        class_id = np.argmax(classes)

        return {"class": self.names[class_id], "class_id": class_id, "score": classes[class_id]}

    def predict(self, image_path):
        tensor = cv2.imread(image_path)
        return self.predict_tensor(tensor)



class HuskyInTwoSteps():
    def __init__(self):
        self.face = FaceAnonymizer()
        self.huskydetector = HuskyLiteDetector()
        self.huskynamer = HuskyNamer()

    def readImage(self, source, display=False):
        image_tensor = cv2.imread(source) if type(source) == str else source
        
        # detector step:
        image_tensor, _ = self.face.detect_face_tensor(image_tensor, blur=3)
        results = self.huskydetector.predict_tensor(image_tensor)
        # using classifier to enhance results
        for res in results:
                box = res["bounding_box"]
                object_tensor = image_tensor[box["top"]:box["bottom"], box["left"]:box["right"]]
                res2 = self.huskynamer.predict_tensor(object_tensor)
                res["class2"], res["score2"] = res2["class"], res["score"]

        # visualize
        for res in results:
                box = res["bounding_box"]
                cv2.rectangle(image_tensor, (box["left"], box["top"]), (box["right"], box["bottom"]), (0,0,255), 3)
                result_text = f"{res['class2']} {res['score2']} (was {res['class']})"
                text_location = (10 + box["left"], 20 + box["top"])
                cv2.putText(image_tensor, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

        if type(source) == str and display:
            cv2.imshow("Husky in 3 steps", image_tensor)
            cv2.waitKey(-1)
        if not display:
            return image_tensor, results
        return image_tensor


class HuskyInTwoSteps():
    def __init__(self):
        self.face = FaceAnonymizer()
        self.huskydetector = HuskyLiteDetector()
        self.huskynamer = HuskyNamer()

    def readTensor(self, source, display=False):
        # image_tensor = cv2.imread(source) if type(source) == str else source
        image_tensor = source
        # detector step:
        image_tensor, _ = self.face.detect_face_tensor(image_tensor, blur=3)
        results = self.huskydetector.predict_tensor(image_tensor)
        # using classifier to enhance results
        for res in results:
                box = res["bounding_box"]
                object_tensor = image_tensor[box["top"]:box["bottom"], box["left"]:box["right"]]
                res2 = self.huskynamer.predict_tensor(object_tensor)
                res["class2"], res["score2"] = res2["class"], res["score"]

        # visualize
        for res in results:
                box = res["bounding_box"]
                cv2.rectangle(image_tensor, (box["left"], box["top"]), (box["right"], box["bottom"]), (0,0,255), 3)
                result_text = f"{res['class2']} {res['score2']} (was {res['class']})"
                text_location = (10 + box["left"], 20 + box["top"])
                cv2.putText(image_tensor, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

        if type(source) == str and display:
            cv2.imshow("Husky in 3 steps", image_tensor)
            cv2.waitKey(-1)
        if not display:
            return image_tensor, results
        return image_tensor


global detector

def init():
    global detector
    detector = HuskyInTwoSteps()



def run(request):
    # print(request)

    # input values
    reqjson = request if type(request) == dict else json.loads(request)
    face_blur = int(reqjson["faceblur"]) if "faceblur" in reqjson else 3
    threshold = float(reqjson["threshold"]) if "threshold" in reqjson else 0.3

    image_bytes = reqjson["img"]
    jpg_as_np = np.frombuffer(b64decode(image_bytes), dtype=np.uint8)
    tensor = cv2.imdecode(jpg_as_np, 3) 


    img, analysis = detector.readTensor(source=tensor, display=False)

    # image is rendered using base64 (not saved)
    _, frame_buff = cv2.imencode('.jpg', img) 
    b64img = b64encode(frame_buff).decode("UTF-8")
    analysis = json.dumps(analysis, indent=4, default=str)

    print(analysis)


    return {"output":b64img, "analysis":analysis}


def preprocess(input):
    return input


