from pathlib import Path
import cv2
from PIL import Image
import numpy as np

img = cv2.imread(str(Path(__file__).with_name("test.jpg")))

img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

img = cv2.GaussianBlur(img , (17, 17), sigmaX=3.0, sigmaY=3.0)

cv2.imshow("frame", img)

cv2.waitKey(0)
