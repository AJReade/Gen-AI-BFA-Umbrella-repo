import gradio as gr
from PIL import Image
from pathlib import Path
from storage import (
    load_image_sets,
    save_image_set,
    save_result,
    delete_image_set,
    promote_to_example,
    list_local_images,
    list_gallery_urls,
    download_to_local,
    is_dataset_url,
    save_image,
    upload_image,
    make_filename,
    generate_id,
    parse_filename,
    parse_result_filename,
    file_url,
    LOCAL_DATA,
)

EXAMPLES_PREFIX = "examples"
UPLOADS_PREFIX = "user_uploads"

for subdir in ["portraits", "garments", "results"]:
    (LOCAL_DATA / EXAMPLES_PREFIX / subdir).mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA / UPLOADS_PREFIX / subdir).mkdir(parents=True, exist_ok=True)


def _load_examples():
    return load_image_sets(EXAMPLES_PREFIX)


def _load_uploads():
    return load_image_sets(UPLOADS_PREFIX)


def _gallery_images(prefix, subdir):
    return list_gallery_urls(prefix, subdir)


def build_demo(process_fn, detect_fn=None):

    def process_and_save(portrait_path, garment_path, category, selected_people):
        if portrait_path is None or garment_path is None:
            raise gr.Error("Please select a portrait and a garment.")
        result = process_fn(portrait_path, garment_path, category, selected_people or None)
        if result and portrait_path and garment_path:
            p_parsed = parse_filename(Path(portrait_path).name)
            g_parsed = parse_filename(Path(garment_path).name)
            if p_parsed and g_parsed:
                for p in [portrait_path, garment_path]:
                    local = Path(p)
                    if local.exists() and str(local).startswith(str(LOCAL_DATA)):
                        upload_image(local, str(local.relative_to(LOCAL_DATA)))
                save_result(UPLOADS_PREFIX, p_parsed["id"], g_parsed["id"], category, result)
        return result

    with gr.Blocks(title="Multi-Person Virtual Try-On") as demo:
        with gr.Tabs():
            # ---- Main VTON Tab ----
            with gr.Tab("Virtual Try-On"):
                gr.Markdown("# Multi-Person Virtual Try-On")
                gr.Markdown("Select a portrait and garment, or upload new ones.")

                selected_portrait = gr.State(value=None)
                selected_garment = gr.State(value=None)

                # User uploads section
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Portraits")
                        portrait_gallery = gr.Gallery(
                            value=_gallery_images(UPLOADS_PREFIX, "portraits"),
                            label="Uploaded Portraits",
                            columns=4,
                            height=200,
                            allow_preview=False,
                        )
                        with gr.Accordion("Upload new portrait", open=False):
                            portrait_upload = gr.Image(type="pil", label="Upload Portrait", sources=["upload", "webcam"])
                            with gr.Row():
                                portrait_url_input = gr.Textbox(label="Or paste image URL", scale=4)
                                portrait_url_btn = gr.Button("Load", size="sm", scale=1)

                    with gr.Column():
                        gr.Markdown("### Garments")
                        garment_gallery = gr.Gallery(
                            value=_gallery_images(UPLOADS_PREFIX, "garments"),
                            label="Uploaded Garments",
                            columns=4,
                            height=200,
                            allow_preview=False,
                        )
                        with gr.Accordion("Upload new garment", open=False):
                            garment_upload = gr.Image(type="pil", label="Upload Garment", sources=["upload", "webcam"])
                            with gr.Row():
                                garment_url_input = gr.Textbox(label="Or paste image URL", scale=4)
                                garment_url_btn = gr.Button("Load", size="sm", scale=1)

                gr.Markdown("### Selected")
                with gr.Row():
                    preview_portrait = gr.Image(label="Selected Portrait", interactive=False, height=250)
                    preview_garment = gr.Image(label="Selected Garment", interactive=False, height=250)

                category = gr.Radio(
                    choices=["tops", "bottoms", "one-pieces"],
                    value="tops",
                    label="Category",
                )

                detect_btn = gr.Button("Detect People", variant="secondary")
                detect_status = gr.Textbox(interactive=False, show_label=False, value="")
                people_gallery = gr.Gallery(
                    label="Detected People",
                    columns=6,
                    height=300,
                    allow_preview=False,
                )
                people_selection = gr.CheckboxGroup(
                    label="Select people to dress",
                    choices=[],
                )

                submit_btn = gr.Button("Try On All", variant="primary")
                result_image = gr.Image(type="pil", label="Result")

                # Examples section
                gr.Markdown("---")
                gr.Markdown("### Examples")
                with gr.Row():
                    ex_portrait_gallery = gr.Gallery(
                        label="Example Portraits",
                        columns=4,
                        height=200,
                        allow_preview=False,
                    )
                    ex_garment_gallery = gr.Gallery(
                        label="Example Garments",
                        columns=4,
                        height=200,
                        allow_preview=False,
                    )
                    ex_result_gallery = gr.Gallery(
                        label="Example Results",
                        columns=4,
                        height=200,
                        allow_preview=False,
                    )
                refresh_examples_btn = gr.Button("Refresh Examples", size="sm")

                # -- Event handlers --

                def on_gallery_select(evt: gr.SelectData):
                    path = evt.value["image"]["path"]
                    local_path = download_to_local(path)
                    return local_path, local_path

                def reset_detection():
                    return (
                        "",
                        [],
                        gr.update(choices=[], value=[]),
                        gr.update(value="Try On All"),
                    )

                detection_reset_outputs = [detect_status, people_gallery, people_selection, submit_btn]

                portrait_gallery.select(
                    on_gallery_select, outputs=[selected_portrait, preview_portrait]
                ).then(reset_detection, outputs=detection_reset_outputs)
                garment_gallery.select(on_gallery_select, outputs=[selected_garment, preview_garment])
                ex_portrait_gallery.select(
                    on_gallery_select, outputs=[selected_portrait, preview_portrait]
                ).then(reset_detection, outputs=detection_reset_outputs)
                ex_garment_gallery.select(on_gallery_select, outputs=[selected_garment, preview_garment])

                def on_portrait_upload(img, current_category):
                    if img is None:
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, None
                    item_id = generate_id()
                    fname = make_filename(item_id, current_category, "portrait")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), path, path, None

                def on_garment_upload(img, current_category):
                    if img is None:
                        return _gallery_images(UPLOADS_PREFIX, "garments"), None, None, None
                    item_id = generate_id()
                    fname = make_filename(item_id, current_category, "garment")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    return _gallery_images(UPLOADS_PREFIX, "garments"), path, path, None

                def on_portrait_url(url, current_category):
                    if not url or not url.strip():
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, ""
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, current_category, "portrait")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), local_path, local_path, ""

                def on_garment_url(url, current_category):
                    if not url or not url.strip():
                        return _gallery_images(UPLOADS_PREFIX, "garments"), None, None, ""
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, current_category, "garment")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    return _gallery_images(UPLOADS_PREFIX, "garments"), local_path, local_path, ""

                portrait_upload.change(
                    on_portrait_upload,
                    inputs=[portrait_upload, category],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_upload],
                ).then(reset_detection, outputs=detection_reset_outputs)
                garment_upload.change(
                    on_garment_upload,
                    inputs=[garment_upload, category],
                    outputs=[garment_gallery, selected_garment, preview_garment, garment_upload],
                )
                portrait_url_btn.click(
                    on_portrait_url,
                    inputs=[portrait_url_input, category],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_url_input],
                ).then(reset_detection, outputs=detection_reset_outputs)
                garment_url_btn.click(
                    on_garment_url,
                    inputs=[garment_url_input, category],
                    outputs=[garment_gallery, selected_garment, preview_garment, garment_url_input],
                )

                def on_detect(portrait_path):
                    if detect_fn is None or portrait_path is None:
                        raise gr.Error("Please select a portrait first.")
                    people = detect_fn(portrait_path)
                    n = len(people)
                    choices = [f"Person {i+1}" for i in range(n)]
                    return (
                        f"Found {n} {'person' if n == 1 else 'people'}",
                        people,
                        gr.update(choices=choices, value=choices),
                        gr.update(value="Try On Selected"),
                    )

                detect_btn.click(
                    lambda: "Detecting people...",
                    outputs=[detect_status],
                ).then(
                    on_detect,
                    inputs=[selected_portrait],
                    outputs=[detect_status, people_gallery, people_selection, submit_btn],
                )

                submit_btn.click(
                    process_and_save,
                    inputs=[selected_portrait, selected_garment, category, people_selection],
                    outputs=result_image,
                )

                def refresh_examples():
                    return (
                        _gallery_images(EXAMPLES_PREFIX, "portraits"),
                        _gallery_images(EXAMPLES_PREFIX, "garments"),
                        _gallery_images(EXAMPLES_PREFIX, "results"),
                    )

                refresh_examples_btn.click(
                    refresh_examples,
                    outputs=[ex_portrait_gallery, ex_garment_gallery, ex_result_gallery],
                )
                demo.load(
                    refresh_examples,
                    outputs=[ex_portrait_gallery, ex_garment_gallery, ex_result_gallery],
                )

            # ---- Admin Tab ----
            with gr.Tab("Admin - Manage Examples"):
                gr.Markdown("## Upload Example Set")
                gr.Markdown("Upload a portrait, garment, and result image together as an example.")

                with gr.Row():
                    with gr.Column():
                        admin_portrait = gr.Image(type="pil", label="Portrait", sources=["upload", "webcam"])
                        admin_portrait_url = gr.Textbox(label="Or paste URL")
                    with gr.Column():
                        admin_garment = gr.Image(type="pil", label="Garment", sources=["upload", "webcam"])
                        admin_garment_url = gr.Textbox(label="Or paste URL")
                    with gr.Column():
                        admin_result = gr.Image(type="pil", label="Result", sources=["upload", "webcam"])
                        admin_result_url = gr.Textbox(label="Or paste URL")

                admin_category = gr.Radio(
                    choices=["tops", "bottoms", "one-pieces"],
                    value="tops",
                    label="Category",
                )

                admin_upload_btn = gr.Button("Add Example", variant="primary")
                admin_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Current Examples")
                admin_examples_table = gr.Dataframe(
                    headers=["ID", "Category", "Portrait", "Garment", "Result"],
                    label="Examples",
                    interactive=False,
                )

                with gr.Row():
                    delete_id = gr.Textbox(label="Example ID to delete", scale=3)
                    delete_category = gr.Textbox(label="Category", scale=2)
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)

                def get_examples_table():
                    examples = _load_examples()
                    rows = []
                    for ex in examples:
                        rows.append([
                            ex["id"],
                            ex["category"],
                            Path(ex["portrait"]).name if ex.get("portrait") else "",
                            Path(ex["garment"]).name if ex.get("garment") else "",
                            Path(ex["result"]).name if ex.get("result") else "",
                        ])
                    return rows

                def _resolve_image(img, url):
                    """Use uploaded image if available, otherwise download from URL."""
                    if img is not None:
                        return img
                    if url and url.strip():
                        local = download_to_local(url.strip())
                        return Image.open(local)
                    return None

                def on_admin_upload(portrait, garment, result, p_url, g_url, r_url, cat):
                    portrait = _resolve_image(portrait, p_url)
                    garment = _resolve_image(garment, g_url)
                    result = _resolve_image(result, r_url)
                    if portrait is None or garment is None:
                        return "Please provide at least portrait and garment (file or URL).", get_examples_table()
                    save_image_set(EXAMPLES_PREFIX, portrait, garment, cat, result)
                    return "Example added.", get_examples_table()

                def on_admin_delete(ex_id, cat):
                    if not ex_id or not cat:
                        return "Please provide both ID and category.", get_examples_table()
                    delete_image_set(EXAMPLES_PREFIX, ex_id.strip(), cat.strip())
                    return "Deleted.", get_examples_table()

                admin_upload_btn.click(
                    on_admin_upload,
                    inputs=[admin_portrait, admin_garment, admin_result,
                            admin_portrait_url, admin_garment_url, admin_result_url,
                            admin_category],
                    outputs=[admin_status, admin_examples_table],
                )
                delete_btn.click(
                    on_admin_delete,
                    inputs=[delete_id, delete_category],
                    outputs=[admin_status, admin_examples_table],
                )

                # ---- Promote from Uploads ----
                gr.Markdown("---")
                gr.Markdown("## Promote from Uploads")
                gr.Markdown("Select a result to promote. The matching portrait and garment are found automatically.")

                promote_portrait = gr.State(value=None)
                promote_garment = gr.State(value=None)
                promote_result = gr.State(value=None)
                promote_category = gr.State(value="tops")

                promo_result_gallery = gr.Gallery(
                    value=_gallery_images(UPLOADS_PREFIX, "results"),
                    label="Results",
                    columns=4,
                    height=200,
                    allow_preview=False,
                )

                with gr.Row():
                    promo_preview_portrait = gr.Image(label="Portrait", interactive=False, height=150)
                    promo_preview_garment = gr.Image(label="Garment", interactive=False, height=150)
                    promo_preview_result = gr.Image(label="Result", interactive=False, height=150)

                promote_btn = gr.Button("Promote to Example", variant="primary")
                promote_status = gr.Textbox(label="Status", interactive=False)

                def on_result_select(evt: gr.SelectData):
                    path = evt.value["image"]["path"]
                    result_local = download_to_local(path)
                    parsed = parse_result_filename(Path(result_local).name)
                    if not parsed:
                        parsed = parse_result_filename(Path(path).name)
                    if not parsed:
                        return None, None, result_local, "tops", None, None, result_local
                    cat = parsed["category"]
                    try:
                        portrait_url = file_url(f"{UPLOADS_PREFIX}/portraits/{make_filename(parsed['portrait_id'], cat, 'portrait')}")
                        garment_url = file_url(f"{UPLOADS_PREFIX}/garments/{make_filename(parsed['garment_id'], cat, 'garment')}")
                        portrait_local = download_to_local(portrait_url)
                        garment_local = download_to_local(garment_url)
                    except Exception:
                        gr.Warning("Matching portrait/garment not found in dataset.")
                        return None, None, result_local, cat, None, None, result_local
                    return portrait_local, garment_local, result_local, cat, portrait_local, garment_local, result_local

                promo_result_gallery.select(
                    on_result_select,
                    outputs=[promote_portrait, promote_garment, promote_result, promote_category,
                             promo_preview_portrait, promo_preview_garment, promo_preview_result],
                )

                def on_promote(portrait_path, garment_path, result_path, cat):
                    if not portrait_path or not garment_path:
                        return "Could not find matching portrait/garment.", get_examples_table()
                    item_id = promote_to_example(portrait_path, garment_path, cat, result_path)
                    return f"Promoted as example {item_id}.", get_examples_table()

                promote_btn.click(
                    on_promote,
                    inputs=[promote_portrait, promote_garment, promote_result, promote_category],
                    outputs=[promote_status, admin_examples_table],
                )

                refresh_promo_btn = gr.Button("Refresh Results", size="sm")
                refresh_promo_btn.click(
                    lambda: _gallery_images(UPLOADS_PREFIX, "results"),
                    outputs=[promo_result_gallery],
                )

                demo.load(get_examples_table, outputs=[admin_examples_table])

    return demo


if __name__ == "__main__":
    def dummy_process(portrait, garment, category, selected_people=None):
        return Image.new("RGB", (512, 512), (200, 200, 200))
    demo = build_demo(dummy_process)
    demo.launch()
