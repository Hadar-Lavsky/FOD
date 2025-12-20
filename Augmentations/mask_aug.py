import numpy as np
from PIL import Image, ImageDraw

img_path = "images/image_1.avif" 
original_img = Image.open(img_path)
width, height = original_img.size

x_center, y_center = 0.5, 0.75  
box_w, box_h = 0.04, 0.03       

x1 = int((x_center - box_w/2) * width)
y1 = int((y_center - box_h/2) * height)
x2 = int((x_center + box_w/2) * width)
y2 = int((y_center + box_h/2) * height)

mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([x1, y1, x2, y2], fill=255)

mask.save("mask.png")

class_id = 0 
with open("label.txt", "w") as f:
    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

print(f"Mask saved as mask.png | Label saved as label.txt")
print(f"Coordinates: Center({x_center}, {y_center}), Size({box_w}, {box_h})")