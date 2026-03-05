import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

UPLOAD_PORTRAITS = Path("data/user_uploads/portraits")
UPLOAD_GARMENTS = Path("data/user_uploads/garments")
UPLOAD_RESULTS = Path("data/user_uploads/results")
EXAMPLE_PORTRAITS = Path("data/examples/portraits")
EXAMPLE_GARMENTS = Path("data/examples/garments")
EXAMPLE_RESULTS = Path("data/examples/results")
UPLOAD_PORTRAITS.mkdir(parents=True, exist_ok=True)
UPLOAD_GARMENTS.mkdir(parents=True, exist_ok=True)
UPLOAD_RESULTS.mkdir(parents=True, exist_ok=True)
EXAMPLE_RESULTS.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

EXAMPLES = [
    {
        "portrait": "data/examples/portraits/Young-Men-Portrait-Smiling.webp",
        "garment": "data/examples/garments/garment.jpg",
        "category": "tops",
        "result": "data/examples/results/Young-Men-Portrait-Smiling_garment_tops.jpg",
    },
]

def list_images(directory):
    d = Path(directory)
    if not d.exists():
        return []
    return sorted([str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def result_filename(portrait_path, garment_path, category):
    p = Path(portrait_path).stem
    g = Path(garment_path).stem
    return f"{p}_{g}_{category}.jpg"

def save_upload(img, upload_dir):
    if img is None:
        return None, list_images(upload_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = upload_dir / f"{ts}.jpg"
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path, "JPEG", quality=85)
    return str(path), list_images(upload_dir)

def on_portrait_upload(img):
    path, gallery = save_upload(img, UPLOAD_PORTRAITS)
    return gallery, path, path, None

def on_garment_upload(img):
    path, gallery = save_upload(img, UPLOAD_GARMENTS)
    return gallery, path, path, None

def on_portrait_select(evt: gr.SelectData):
    path = evt.value["image"]["path"]
    return path, path

def on_garment_select(evt: gr.SelectData):
    path = evt.value["image"]["path"]
    return path, path

def build_demo(process_fn):
    def process_and_save(portrait_path, garment_path, category):
        result = process_fn(portrait_path, garment_path, category)
        if result and portrait_path and garment_path:
            name = result_filename(portrait_path, garment_path, category)
            result.save(UPLOAD_RESULTS / name, "JPEG", quality=85)
        return result

    with gr.Blocks(title="Multi-Person Virtual Try-On") as demo:
        gr.Markdown("# Multi-Person Virtual Try-On")
        gr.Markdown("Select a portrait and garment from your uploads, or upload new ones.")

        selected_portrait = gr.State(value=None)
        selected_garment = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Portraits")
                portrait_gallery = gr.Gallery(
                    value=list_images(UPLOAD_PORTRAITS),
                    label="Uploaded Portraits",
                    columns=4,
                    height=200,
                    allow_preview=False,
                )
                with gr.Accordion("Upload new portrait", open=False):
                    portrait_upload = gr.Image(type="pil", label="Upload Portrait")

            with gr.Column():
                gr.Markdown("### Garments")
                garment_gallery = gr.Gallery(
                    value=list_images(UPLOAD_GARMENTS),
                    label="Uploaded Garments",
                    columns=4,
                    height=200,
                    allow_preview=False,
                )
                with gr.Accordion("Upload new garment", open=False):
                    garment_upload = gr.Image(type="pil", label="Upload Garment")

        gr.Markdown("### Selected")
        with gr.Row():
            preview_portrait = gr.Image(label="Selected Portrait", interactive=False, height=250)
            preview_garment = gr.Image(label="Selected Garment", interactive=False, height=250)

        category = gr.Radio(
            choices=["tops", "bottoms", "one-pieces"],
            value="tops",
            label="Category",
        )

        submit_btn = gr.Button("Try On", variant="primary")
        result_image = gr.Image(type="pil", label="Result")

        # Examples gallery
        gr.Markdown("---")
        gr.Markdown("### Examples")
        for ex in EXAMPLES:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(value=ex["portrait"], label="Portrait", interactive=False, height=200)
                    use_portrait_btn = gr.Button("Use portrait", size="sm")
                    use_portrait_btn.click(
                        lambda p=ex["portrait"]: (p, p),
                        outputs=[selected_portrait, preview_portrait],
                    )
                with gr.Column(scale=1):
                    gr.Image(value=ex["garment"], label="Garment", interactive=False, height=200)
                    use_garment_btn = gr.Button("Use garment", size="sm")
                    use_garment_btn.click(
                        lambda g=ex["garment"]: (g, g),
                        outputs=[selected_garment, preview_garment],
                    )
                with gr.Column(scale=0, min_width=80):
                    gr.Markdown(f"**{ex['category']}**")
                with gr.Column(scale=1):
                    result_path = ex["result"]
                    if Path(result_path).exists():
                        gr.Image(value=result_path, label="Result", interactive=False, height=200)
                    else:
                        gr.Markdown("*Result not yet generated*")

        # Events
        portrait_gallery.select(on_portrait_select, outputs=[selected_portrait, preview_portrait])
        garment_gallery.select(on_garment_select, outputs=[selected_garment, preview_garment])

        portrait_upload.change(
            on_portrait_upload,
            inputs=portrait_upload,
            outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_upload],
        )
        garment_upload.change(
            on_garment_upload,
            inputs=garment_upload,
            outputs=[garment_gallery, selected_garment, preview_garment, garment_upload],
        )

        submit_btn.click(
            process_and_save,
            inputs=[selected_portrait, selected_garment, category],
            outputs=result_image,
        )

    return demo

if __name__ == "__main__":
    def dummy_process(portrait, garment, category):
        return Image.new("RGB", (512, 512), (200, 200, 200))
    demo = build_demo(dummy_process)
    demo.launch()
