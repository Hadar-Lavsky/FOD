import numpy as np
from PIL import Image, ImageDraw
import os

# נתיב התמונה
img_path = "images/image_1.avif" 
original_img = Image.open(img_path)
width, height = original_img.size

# --- הגדרות מיקום וגודל ---
x_center, y_center = 0.5, 0.75  # מיקום במרכז המסלול (לפי הדגימה הקודמת שלך)

# הגדלנו את הערכים פי 3 מהמקור (0.04 -> 0.12 | 0.03 -> 0.09)
box_w, box_h = 0.12, 0.09  

# חישוב קואורדינטות בפיקסלים
x1 = int((x_center - box_w/2) * width)
y1 = int((y_center - box_h/2) * height)
x2 = int((x_center + box_w/2) * width)
y2 = int((y_center + box_h/2) * height)

# יצירת ה-Mask (שחור עם מלבן לבן)
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([x1, y1, x2, y2], fill=255)

# שמירה בתיקיית images כדי שהסקריפט השני ימצא אותו
mask_output_path = "images/mask.png"
mask.save(mask_output_path)

# שמירת לייבל בפורמט YOLO (כדי שיתאים למודל ה-YOLOv12 שלך)
class_id = 1  # שיניתי ל-1 עבור כתם שמן (FOD)
with open("label.txt", "w") as f:
    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

print(f"✓ Mask enlarged and saved to: {mask_output_path}")
print(f"✓ New Size: {box_w*100}% width, {box_h*100}% height of image")