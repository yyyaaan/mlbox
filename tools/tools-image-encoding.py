"""
The code snippets below provides BASE64 image solution
Base64 encoded can be displayed in HTML directly, and save as string
If request parse image as b64, use corresponding library to encode
For decoding, standard b64decode is always necessary
"""



# encode_from_file(file_path):
from base64 import urlsafe_b64encode
file_path = '/Users/pannnyan/Documents/DevGit/transfer-learning/c.jpg'
the_bytes = open(file_path, 'rb').read() # bytes
the_b64 = urlsafe_b64encode(the_bytes) # base64, may need .decode("UTF-8")


# decode using tensorflow only, file format needs to be determined
import tensorflow as tf
decoded_bytes = tf.io.decode_base64(the_b64)
decoded_tftensor = tf.io.decode_jpeg(decoded_bytes) 
decoded_tensor = decoded_tftensor.numpy()

# decode using OpenCV
from base64 import urlsafe_b64decode
import numpy as np
import cv2
decoded_bytes2 = urlsafe_b64decode(the_b64)
decoded_numpy2 = np.frombuffer(decoded_bytes2, dtype=np.uint8) # not properly shaped
decoded_tensor2 = cv2.imdecode(decoded_numpy2, 3) 


# encode to base64 for display
from base64 import b64encode
_, jpgbytes = cv2.imencode('.jpg', decoded_tensor2)  # cv2 method
jpgbytes = tf.io.encode_jpeg(decoded_tftensor).numpy() # tensorflow
b64img = b64encode(jpgbytes).decode("UTF-8") # this always the same

# test the output in browser
with open("tmp.html", "w") as f:
    f.write(f'<html><img src="data:image/jpeg;base64, {b64img}"></img>=</html>')
        
