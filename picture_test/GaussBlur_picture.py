from pathlib import Path
import cv2
from PIL import Image
import numpy as np

img = cv2.imread(r"C:\Users\20147\Documents\Codex\2026-06-26\https-www-zhihu-com-question-326034346\outputs\normalization_learning\picture_test\test.jpg")

img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

img = cv2.GaussianBlur(img , (17, 17), sigmaX=3.0, sigmaY=3.0)

cv2.imshow("frame", img)

cv2.waitKey(0)