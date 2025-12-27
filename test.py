import cv2
import numpy as np
from PIL import Image
img_path = r"C:\Users\surya\Downloads\tanmay_1743333287327_1743333298865.avif"
# var = img_path.split(".")[0]
# var += ".jpeg"
# temp = cv2.imread(img_path)
# print(type(temp))

pil_img = Image.open(img_path).convert('RGB')
img_np = np.array(pil_img)
img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
print(type(img_cv))
cv2.imwrite("output.jpeg", img_cv)