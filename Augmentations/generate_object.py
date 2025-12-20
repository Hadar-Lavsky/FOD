import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import os

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

def process_and_label(image_path, mask_path, prompt, class_id, output_dir="output"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")
    
    generated_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_path = os.path.join(output_dir, f"{base_name}_gen.jpg")
    generated_image.save(img_save_path)
    

    mask_np = np.array(mask_image.convert("L"))
    rows, cols = np.where(mask_np > 127) 
    
    if len(rows) == 0: return "No mask detected"
    
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    

    img_w, img_h = init_image.size
    
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    

    label_path = os.path.join(output_dir, f"{base_name}_gen.txt")
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return f"Success: Saved to {output_dir}"

process_and_label("image/mask.png", "image/image_1.avif", "a metallic bolt on the runway, realistic shadow", class_id=0)