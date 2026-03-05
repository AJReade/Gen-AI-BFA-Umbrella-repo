import spaces
import cv2
import numpy as np
from PIL import Image
import torch
from fashn_vton import TryOnPipeline
from ultralytics import YOLO
import gradio as gr
from pathlib import Path
import subprocess
import sys
from scipy.spatial import cKDTree
from ui import build_demo


class MultiPersonVTON:
    def __init__(self, weights_dir="./weights"):
        print("Initializing Multi-Person VTON pipeline...")
        self.pipeline = TryOnPipeline(weights_dir=weights_dir)
        self.model = YOLO("yolo26n-seg.pt")
        print("Pipeline initialized")

    def get_mask(self, result, H, W):
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
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
        img_np = np.array(img) if isinstance(img, Image.Image) else img.copy()
        people = []
        for mask in masks:
            cutout = img_np.copy()
            cutout[~mask] = 255
            people.append(Image.fromarray(cutout))
        return people

    def apply_vton_to_people(self, people, garment, category="tops"):
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
        vton_masks = []
        for people in vton_people:
            people_arr = np.array(people)
            gray = cv2.cvtColor(people_arr, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            mask = mask.astype(bool)
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_u8 = (mask_clean.astype(np.uint8) * 255)
            mask_blur = cv2.GaussianBlur(mask_u8, (3, 3), 1)
            vton_masks.append(mask_blur)
        return vton_masks

    def contour_curvature(self, contour, k=5):
        pts = contour[:, 0, :].astype(np.float32)
        N = len(pts)
        curv = np.zeros(N)
        for i in range(N):
            p_prev = pts[(i - k) % N]
            p = pts[i]
            p_next = pts[(i + k) % N]
            v1 = p - p_prev
            v2 = p_next - p
            v1 /= (np.linalg.norm(v1) + 1e-6)
            v2 /= (np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            curv[i] = angle
        return curv

    def frontness_score(self, mask_a, mask_b):
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
        inter_pts = np.column_stack(np.where(inter))[:, ::-1]
        tree_a = cKDTree(ca[:, 0, :])
        tree_b = cKDTree(cb[:, 0, :])
        _, idx_a = tree_a.query(inter_pts, k=1)
        _, idx_b = tree_b.query(inter_pts, k=1)
        score_a = curv_a[idx_a].mean()
        score_b = curv_b[idx_b].mean()
        return score_a - score_b

    def estimate_front_to_back_order(self, masks):
        n = len(masks)
        scores = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                scores[i] += self.frontness_score(masks[i], masks[j])
        order = np.argsort(-scores)
        return order, scores

    def remove_original_people(self, image, person_masks):
        image_np = np.array(image)
        combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for mask in person_masks:
            combined_mask[mask] = 255
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        inpainted = cv2.inpaint(image_np, combined_mask, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted), combined_mask

    def clean_vton_edges_on_overlap(self, img_pil, mask_uint8, other_masks_uint8,
                                    erode_iters=1, edge_dilate=2, inner_erode=2):
        src = np.array(img_pil).copy()
        others_union = np.zeros_like(mask_uint8, dtype=np.uint8)
        for m in other_masks_uint8:
            others_union = np.maximum(others_union, m)
        overlap = (mask_uint8 > 0) & (others_union > 0)
        overlap = overlap.astype(np.uint8) * 255
        if overlap.sum() == 0:
            return img_pil, mask_uint8
        kernel = np.ones((3, 3), np.uint8)
        tight_mask = cv2.erode(mask_uint8, kernel, iterations=erode_iters)
        edge = cv2.Canny(tight_mask, 50, 150)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=edge_dilate)
        overlap_band = cv2.dilate(overlap, np.ones((5, 5), np.uint8), iterations=1)
        edge = cv2.bitwise_and(edge, overlap_band)
        if edge.sum() == 0:
            return img_pil, tight_mask
        inner = cv2.erode(tight_mask, np.ones((5, 5), np.uint8), iterations=inner_erode)
        inner_rgb = cv2.inpaint(src, 255 - inner, 3, cv2.INPAINT_TELEA)
        src[edge > 0] = inner_rgb[edge > 0]
        return Image.fromarray(src), tight_mask

    def clean_masks(self, vton_people, vton_masks):
        cleaned_vton_people = []
        cleaned_vton_masks = []
        for i in range(len(vton_people)):
            other_masks = [m for j, m in enumerate(vton_masks) if j != i]
            cleaned_img, cleaned_mask = self.clean_vton_edges_on_overlap(
                vton_people[i], vton_masks[i], other_masks,
                erode_iters=1, edge_dilate=2, inner_erode=2
            )
            cleaned_vton_people.append(cleaned_img)
            cleaned_vton_masks.append(cleaned_mask)
        return cleaned_vton_people, cleaned_vton_masks

    def process_group_image(self, group_image, garment_image, category="tops"):
        print("Step 1: Loading images...")
        if isinstance(group_image, np.ndarray):
            group_image = Image.fromarray(group_image)
        if isinstance(garment_image, np.ndarray):
            garment_image = Image.fromarray(garment_image)
        if isinstance(group_image, Image.Image):
            group_image.save("people.png")

        img_bgr = cv2.imread("people.png")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        print("Step 2: Getting segmentation masks with YOLO...")
        results = self.model("people.png")
        result = results[0]
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
        img = cv2.resize(img, vton_people[0].size)

        print("Step 7: Creating clean background by removing original people...")
        clean_background, person_mask = self.remove_original_people(img, masks)
        clean_background_np = np.array(clean_background)

        print("Step 8: Recomposing final image...")
        recomposed = clean_background_np.copy()
        for i in order:
            vton_mask = cleaned_vton_masks[i]
            img_pil = cleaned_vton_people[i]
            out = recomposed.astype(np.float32)
            src = np.array(img_pil).astype(np.float32)
            alpha = (vton_mask.astype(np.float32) / 255.0)[..., None]
            src = src * alpha
            out = src + (1 - alpha) * out
            recomposed = out.astype(np.uint8)

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

ensure_weights()

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiPersonVTON()
    return _pipeline

@spaces.GPU
def process_images(selected_portrait, selected_garment, category):
    if selected_portrait is None or selected_garment is None:
        raise gr.Error("Please select a portrait and a garment.")
    portrait = Image.open(selected_portrait) if isinstance(selected_portrait, str) else selected_portrait
    garment = Image.open(selected_garment) if isinstance(selected_garment, str) else selected_garment
    pipeline = get_pipeline()
    new_width = 576
    w, h = portrait.size
    new_height = int(h * new_width / w)
    resized = portrait.resize((new_width, new_height), Image.LANCZOS)
    result, _ = pipeline.process_group_image(resized, garment, category)
    return result

demo = build_demo(process_images)
demo.launch()
