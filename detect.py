import tensorflow as tf
import numpy as np

from models.clshuskydetector import HuskyDetector
from models.clshuskynamer import HuskyNamer

huskydetector = HuskyDetector()
huskynamer = HuskyNamer()


###########def tmp_encoder():
from base64 import urlsafe_b64encode
file_path = './data/aa.jpg'
the_bytes = open(file_path, 'rb').read() # bytes
the_b64 = urlsafe_b64encode(the_bytes) # base64, may need .decode("UTF-8")
###########


# from b64 tensor
tensor = tf.io.decode_jpeg(tf.io.decode_base64(the_b64))

# using classifier to enhance results from detector
results = huskydetector.detect(tensor)

for res in results:
    box = res["bounding_box"]
    object_tensor = tensor[box["top"]:box["bottom"], box["left"]:box["right"]]
    res2 = huskynamer.predict(object_tensor)
    res["class2"], res["score2"] = res2["class"], res["score"]

# visualize
boxes = np.asanyarray([res['box'] for res in results])
output_tensor = tf.image.draw_bounding_boxes(
    np.expand_dims(tensor, axis=0), 
    np.expand_dims(boxes, axis=0),
    np.array([[0.5, 0.8, 0.0]]))
output_tensor = np.array(output_tensor[0,:,:,:], dtype=np.uint8)




########## encode to base64 for display
from base64 import b64encode
jpgbytes = tf.io.encode_jpeg(output_tensor).numpy() # tensorflow
b64img = b64encode(jpgbytes).decode("UTF-8") # this always the same

# test the output in browser
with open("tmp.html", "w") as f:
    f.write(f'<html><img width="70%" src="data:image/jpeg;base64, {b64img}"></img>=</html>')
#############

