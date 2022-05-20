
from os.path import exists
import cv2

# face detetor
class FaceHaarDetector:
    def __init__(self, modelfile="facehaar.xml", threshold=0.2):
        modelpath = modelfile if exists(modelfile) else f"./models/{modelfile}"
        self.face_cascade = cv2.CascadeClassifier(modelpath)
    
    def detect(self, tensor, blur=0):
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