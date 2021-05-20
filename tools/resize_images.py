import cv2
import os
import shutil
import sys

dir = r'C:\Users\79233\Desktop\images'
output_dir = os.path.join(r'C:\Users\79233\Desktop', 'resize')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

images = os.listdir(dir)

for i in range(len(images)):
    img_path = os.path.join(dir, images[i])
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (360, 640))
    cv2.imwrite(os.path.join(output_dir, '{:06d}.jpg'.format(i)), img)
