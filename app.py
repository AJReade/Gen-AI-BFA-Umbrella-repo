import spaces
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
from scipy.spatial import cKDTree

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
        for people in vton_people:
          people_arr = np.array(people)
          gray = cv2.cvtColor(people_arr, cv2.COLOR_RGB2GRAY)
          _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
          mask = mask.astype(bool)
        
          # remove noise, fill small holes
          kernel = np.ones((5, 5), np.uint8)
          mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
          # Fill small gaps inside the silhouette
          mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        
          # Light blur to round jagged edges
          mask_u8 = (mask_clean.astype(np.uint8) * 255)
          mask_blur = cv2.GaussianBlur(mask_u8, (3, 3), 1)
          vton_masks.append(mask_blur)
        
        return vton_masks

    def contour_curvature(self, contour, k=5):
        """
        Estimate curvature at each contour point using angle change.
        contour: (N, 1, 2) from cv2.findContours
        returns: (N,) curvature magnitudes
        """
        pts = contour[:, 0, :].astype(np.float32)
        N = len(pts)
    
        curv = np.zeros(N)
        for i in range(N):
            p_prev = pts[(i - k) % N]
            p = pts[i]
            p_next = pts[(i + k) % N]
    
            v1 = p - p_prev
            v2 = p_next - p
    
            # angle between vectors
            v1 /= (np.linalg.norm(v1) + 1e-6)
            v2 /= (np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            curv[i] = angle
    
        return curv
    
    def frontness_score(self, mask_a, mask_b):
        """
        Returns positive if A is likely in front of B.
        """
        inter = mask_a & mask_b
        if inter.sum() < 50:
            return 0.0
    
        cnts_a, _ = cv2.findContours(mask_a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_b, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts_a or not cnts_b:
            return 0.0
    
        ca = max(cnts_a, key=len)
        cb = max(cnts_b, key=len)
    
        curv_a = self.contour_curvature(ca)
        curv_b = self.contour_curvature(cb)
    
        # Points near intersection
        inter_pts = np.column_stack(np.where(inter))[:, ::-1]  # (x, y)
    
        tree_a = cKDTree(ca[:, 0, :])
        tree_b = cKDTree(cb[:, 0, :])
    
        _, idx_a = tree_a.query(inter_pts, k=1)
        _, idx_b = tree_b.query(inter_pts, k=1)
    
        score_a = curv_a[idx_a].mean()
        score_b = curv_b[idx_b].mean()
    
        return score_a - score_b

    def estimate_front_to_back_order(self, masks):
        """
        Returns indices sorted front → back.
        """
        n = len(masks)
        scores = np.zeros(n)
    
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                scores[i] += self.frontness_score(masks[i], masks[j])
    
        order = np.argsort(-scores)  # descending = front first
        return order, scores
    
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

    def clean_vton_edges_on_overlap(self, img_pil, mask_uint8, other_masks_uint8,
                                    erode_iters=1,
                                    edge_dilate=2,
                                    inner_erode=2):
        """
        img_pil: PIL.Image (RGB) of one VTON output
        mask_uint8: uint8 mask (0..255) for this person
        other_masks_uint8: list of uint8 masks (0..255) for other people
        Returns: cleaned PIL.Image, tightened mask (uint8)
        """
        src = np.array(img_pil).copy()
    
        # Union of other people masks
        others_union = np.zeros_like(mask_uint8, dtype=np.uint8)
        for m in other_masks_uint8:
            others_union = np.maximum(others_union, m)
    
        # Where this person overlaps with others
        overlap = (mask_uint8 > 0) & (others_union > 0)
        overlap = overlap.astype(np.uint8) * 255
    
        if overlap.sum() == 0:
            # No overlap → return untouched
            return img_pil, mask_uint8
    
        # 1) Tighten this person's mask slightly
        kernel = np.ones((3, 3), np.uint8)
        tight_mask = cv2.erode(mask_uint8, kernel, iterations=erode_iters)
    
        # 2) Edge band of this person
        edge = cv2.Canny(tight_mask, 50, 150)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=edge_dilate)
    
        # 3) Only keep edge pixels that are near overlap regions
        overlap_band = cv2.dilate(overlap, np.ones((5, 5), np.uint8), iterations=1)
        edge = cv2.bitwise_and(edge, overlap_band)
    
        if edge.sum() == 0:
            return img_pil, tight_mask
    
        # 4) Clean interior
        inner = cv2.erode(tight_mask, np.ones((5, 5), np.uint8), iterations=inner_erode)
    
        # 5) Pull interior colours outward to contaminated overlap edge band
        inner_rgb = cv2.inpaint(src, 255 - inner, 3, cv2.INPAINT_TELEA)
        src[edge > 0] = inner_rgb[edge > 0]
    
        return Image.fromarray(src), tight_mask

    def clean_masks(self, vton_people, vton_masks):
        cleaned_vton_people = []
        cleaned_vton_masks = []
        
        for i in range(len(vton_people)):
            img_pil = vton_people[i]
            mask = vton_masks[i]
        
            other_masks = [m for j, m in enumerate(vton_masks) if j != i]
        
            cleaned_img, cleaned_mask = self.clean_vton_edges_on_overlap(
                img_pil,
                mask,
                other_masks,
                erode_iters=1,
                edge_dilate=2,
                inner_erode=2
            )
        
            cleaned_vton_people.append(cleaned_img)
            cleaned_vton_masks.append(cleaned_mask)

    return cleaned_vton_people, cleaned_vton_masks
    
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
            group_image.save("people.png")
        
        # Load image
        img_bgr = cv2.imread("people.png")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        
        print("Step 2: Getting segmentation masks with YOLO...")
        
        # Run YOLO segmentation
        results = self.model("people.png")
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
        order, scores = self.estimate_front_to_back_order(vton_masks)
        cleaned_vton_people, cleaned_vton_masks = self.clean_masks(vton_people, vton_masks)
        
        print("Step 6: Resizing to match dimensions...")
        # Resize original image to match VTON results (matches notebook)
        img = cv2.resize(img, vton_people[0].size)
        
        print("Step 7: Creating clean background by removing original people...")
        # Remove original people using inpainting
        clean_background, person_mask = self.remove_original_people(img, masks)
        clean_background_np = np.array(clean_background)
        
        print("Step 8: Recomposing final image...")
        # Start with clean background
        recomposed = clean_background_np.copy()
        
        for i in order:
            vton_mask = cleaned_vton_masks[i]              # uint8 0..255
            img_pil = cleaned_vton_people[i]
            out = recomposed.astype(np.float32)
            src = np.array(img_pil).astype(np.float32)
            alpha = (vton_mask.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)
            src = src * alpha
            out = src + (1 - alpha) * out
        
            recomposed = out.astype(np.uint8)
        
        # Convert back to PIL Image
        final_image = Image.fromarray(recomposed)
        
        return final_image, {
            "original": Image.fromarray(img),
            "clean_background": clean_background,
            "person_mask": Image.fromarray(person_mask),
            "num_people": len(people),
            "individual_people": people,
            "vton_results": cleaned_vton_people,
            "masks": masks,
            "vton_masks": cleaned_vton_masks
        }

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


# Download weights at startup (CPU-only, no GPU needed)
ensure_weights()

# Lazy-load pipeline on first GPU request
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiPersonVTON()
    return _pipeline

@spaces.GPU
def process_images(group_img, garment_img, category):
    pipeline = get_pipeline()
    new_width = 576
    w, h = group_img.size
    new_height = int(h * new_width / w)
    resized = group_img.resize((new_width, new_height), Image.LANCZOS)
    result, _ = pipeline.process_group_image(resized, garment_img, category)
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
        ["people.png", "garment.webp", "tops"],
    ],
)

demo.launch()
