#   ____                     _  __     __           ____                _           
#  |  _ \ __ _ ___  ___ __ _| | \ \   / /__   ___  |  _ \ ___  __ _  __| | ___ _ __ 
#  | |_) / _` / __|/ __/ _` | |  \ \ / / _ \ / __| | |_) / _ \/ _` |/ _` |/ _ \ '__|
#  |  __/ (_| \__ \ (_| (_| | |   \ V / (_) | (__  |  _ <  __/ (_| | (_| |  __/ |   
#  |_|   \__,_|___/\___\__,_|_|    \_/ \___/ \___| |_| \_\___|\__,_|\__,_|\___|_|   
                                                                                  
import xml.etree.ElementTree as ET
import cv2
import os

def all_obj_detection_to_classes(folders = ["./huskies_validation", "./huskies_training"], outfolder = "./husky_names/"):
    for d in folders:
      for f in os.listdir(d):
            if f[-3:] == "xml":
                obj_detection_to_classes(f"{d}/{f}", outfolder)

def obj_detection_to_classes(path_xml, outfolder):

    tree = ET.parse(path_xml) 
    root = tree.getroot()
    # height, width, channels = [int(x.text) for x in root.find("size")]

    img = cv2.imread(f"{path_xml[:-3]}jpg")

    for member in root.findall('object'):
        # name and bbox
        name = member[0].text
        x1, y1, x2, y2 = [int(x.text) for x in member[4]]

        # crop and save
        cropped = img[y1:y2, x1:x2]
        try:
            the_folder = f"{outfolder}/{name}/".replace("//","/")
            os.mkdir(the_folder) if not os.path.exists(the_folder) else None
            cv2.imwrite(f"{the_folder}{path_xml.split('/')[-1][:-4]}.jpg", cropped)
        except Exception as e:
                print(str(e))

            