import cv2
import numpy as np
from PIL import Image
import torch
from fashn_vton import TryOnPipeline
from ultralytics import YOLO
import gradio as gr
import requests
from io import BytesIO
from pathlib import Path
import subprocess
import sys

class MultiPersonVTON:
    def __init__(self, weights_dir="./weights"):
        """
        Initialize the Multi-Person Virtual Try-On pipeline
        """
        print("Initializing Multi-Person VTON pipeline...")
        
        # Initialize VTON pipeline
        self.pipeline = TryOnPipeline(weights_dir=weights_dir)
        
        # Initialize YOLO for segmentation
        self.model = YOLO("yolo26n-seg.pt")
        
        print("Pipeline initialized")
    
    def get_mask(self, result, H, W):
        """
        Get person segmentation masks from a YOLO result
        Matches the notebook's get_mask function exactly
        """
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        person_idxs = cls_ids == 0
        person_polygons = [poly for poly, keep in zip(result.masks.xy, person_idxs) if keep]
        masks = []
        for poly in person_polygons:
            mask = np.zeros((H, W), dtype=np.uint8)
            poly_int = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask, [poly_int], 1)
            masks.append(mask.astype(bool))
        return masks
    
    def extract_people(self, img, masks):
        """
        Extract each person from the image using masks
        Matches the notebook's extraction code exactly
        """
        img_np = np.array(img) if isinstance(img, Image.Image) else img.copy()
        
        people = []
        for mask in masks:
            cutout = img_np.copy()
            cutout[~mask] = 255
            img_pil = Image.fromarray(cutout)
            people.append(img_pil)
        
        return people
    
    def apply_vton_to_people(self, people, garment, category="tops"):
        """
        Apply VTON to each extracted person
        Matches the notebook's VTON application exactly
        """
        vton_people = []
        for person in people:
            result = self.pipeline(
                person_image=person,
                garment_image=garment,
                category=category
            )
            vton_people.append(result.images[0])
        
        return vton_people
    
    def get_vton_masks(self, vton_people):
        """
        Get segmentation masks for VTON results
        Matches the notebook's vton_masks generation exactly
        """
        vton_masks = []
        for person in vton_people:
            W, H = person.size
            results = self.model(np.array(person))
            result = results[0]
            mask = self.get_mask(result, H, W)[0]
            vton_masks.append(mask)
        
        return vton_masks
    
    def remove_original_people(self, image, person_masks):
        """
        Remove original people from image using inpainting
        Matches the notebook's remove_original_people function exactly
        """
        image_np = np.array(image)
        
        # Combine all person masks into one
        combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for mask in person_masks:
            combined_mask[mask] = 255
        
        # Dilate mask slightly to catch edges
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        # Use inpainting to fill the masked areas with background
        inpainted = cv2.inpaint(image_np, combined_mask, 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(inpainted), combined_mask
    
    def process_group_image(self, group_image, garment_image, category="tops"):
        """
        Main function to process a group photo with multiple people
        Follows the exact steps from the notebook
        """
        print("Step 1: Loading images...")
        if isinstance(group_image, np.ndarray):
            group_image = Image.fromarray(group_image)
        if isinstance(garment_image, np.ndarray):
            garment_image = Image.fromarray(garment_image)
        
        # Load and prepare the original image
        if isinstance(group_image, Image.Image):
            # Save temporarily to match notebook's file loading pattern
            group_image.save("examples/data/people.png")
        
        # Load image
        img_bgr = cv2.imread("examples/data/people.png")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        
        print("Step 2: Getting segmentation masks with YOLO...")
        
        # Run YOLO segmentation
        results = self.model("examples/data/people.png")
        result = results[0]
        
        # Get masks using notebook's get_mask function
        masks = self.get_mask(result, H, W)
        print(f"Found {len(masks)} people")
        
        print("Step 3: Extracting individual people...")
        people = self.extract_people(img, masks)
        
        print("Step 4: Applying VTON to each person...")
        vton_people = self.apply_vton_to_people(people, garment_image, category)
        
        print("Step 5: Getting masks for VTON results...")
        vton_masks = self.get_vton_masks(vton_people)
        
        print("Step 6: Resizing to match dimensions...")
        # Resize original image to match VTON results (matches notebook)
        img = cv2.resize(img, vton_people[0].size)
        
        print("Step 7: Creating clean background by removing original people...")
        # Remove original people using inpainting
        clean_background, person_mask = self.remove_original_people(img, masks)
        clean_background_np = np.array(clean_background)
        
        print("Step 8: Recomposing final image...")
        # Start with clean background
        recomposed_clean = clean_background_np.copy()
        
        # Overlay VTON results
        for mask, vton_mask, img_pil in zip(masks, vton_masks, vton_people):
            vton_np = np.array(img_pil)
            recomposed_clean[vton_mask] = vton_np[vton_mask]
        
        # Convert back to PIL Image
        final_image = Image.fromarray(recomposed_clean)
        
        return final_image, {
            "original": Image.fromarray(img),
            "clean_background": clean_background,
            "person_mask": Image.fromarray(person_mask),
            "num_people": len(people),
            "individual_people": people,
            "vton_results": vton_people,
            "masks": masks,
            "vton_masks": vton_masks
        }

# Create Gradio interface
def create_demo():
    # Initialize pipeline
    pipeline = MultiPersonVTON()
    
    def process_images(group_img, garment_img, category):
        """Wrapper function for Gradio"""
        result, _ = pipeline.process_group_image(group_img, garment_img, category)
        return result
    
    demo = gr.Interface(
        fn=process_images,
        inputs=[
            gr.Image(type="pil", label="Group Photo"),
            gr.Image(type="pil", label="Garment"),
            gr.Radio(
                choices=["tops", "bottoms", "one-pieces"],
                value="tops",
                label="Category",
            ),
        ],
        outputs=gr.Image(type="pil", label="Result"),
        title="Multi-Person Virtual Try-On",
        description="Upload a group photo and a garment to try it on everyone in the photo!",
        examples=[
            ["examples/data/people.png", "examples/data/garment.webp", "tops"],
        ],
    )
    
    return demo


WEIGHTS_DIR = Path("./weights")

def ensure_weights():
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()):
        print("Weights already present, skipping download.")
        return

    print("Downloading weights...")
    subprocess.check_call([
        sys.executable,
        "fashn-vton-1.5/scripts/download_weights.py",
        "--weights-dir",
        str(WEIGHTS_DIR),
    ])
    

# Run function for Hugging Face Spaces
ensure_weights()
demo = create_demo()
demo.launch(
    sever_port = 7860,
    server_name = "0.0.0.0"
)
