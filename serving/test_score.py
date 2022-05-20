from score import *

image_path = '/Users/pannnyan/Documents/DevGit/transfer-learning/c.jpg'
img = cv2.imread(image_path)
_, frame_buff = cv2.imencode('.jpg', img) 
image_bytes = b64encode(frame_buff).decode("UTF-8")


init()
output = run({"img":image_bytes})




jpg_as_np = np.frombuffer(b64decode(output["output"]), dtype=np.uint8)
tensor = cv2.imdecode(jpg_as_np, 3) 
cv2.imwrite("out.jpg", tensor)


# conda env export > environment_droplet.yml


# encode

the_bytes = open(image_path, 'rb').read() # bytes
the_b64 = b64encode(the_bytes).decode("UTF-8") # base64


