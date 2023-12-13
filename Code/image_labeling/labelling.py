import cv2
import os
import time
import uuid


IMAGES_PATH = 'images/collectedimages'

labels =["Hey","merci","oui","non","amour","je t'aime","peace","A","B","C"]

num_imgs=20

for label in labels:
    label_path = os.path.join(IMAGES_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print( 'Collecting images for {}'.format(label))
    time. sleep(5)
    for imgnum in range(num_imgs):
        ret, frame = cap.read()
        img_name= os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(img_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()