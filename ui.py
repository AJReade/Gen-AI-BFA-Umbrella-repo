import gradio as gr
from PIL import Image
from pathlib import Path
from storage import (
    load_image_sets,
    save_image_set,
    save_multi_result,
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
    parse_multi_result_filename,
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


MAX_PEOPLE = 8


def build_demo(process_fn, detect_fn=None, max_people=MAX_PEOPLE):

    def process_and_save(portrait_path, garment_pool, num_detected, *assignment_args):
        if portrait_path is None:
            raise gr.Error("Please select a portrait.")
        if not garment_pool:
            raise gr.Error("Please add at least one garment to the pool.")
        result = process_fn(portrait_path, garment_pool, num_detected, *assignment_args)
        if result and portrait_path:
            p_parsed = parse_filename(Path(portrait_path).name)
            if p_parsed:
                # Upload portrait if local
                local = Path(portrait_path)
                if local.exists() and str(local).startswith(str(LOCAL_DATA)):
                    upload_image(local, str(local.relative_to(LOCAL_DATA)))
                # Upload garments and build assignment metadata for filename
                n = num_detected if num_detected else 0
                max_p = len(assignment_args) // 2
                pool_by_label = {g["label"]: g for g in garment_pool}
                file_assignments = []
                for i in range(n):
                    dd_val = assignment_args[i]
                    cat_val = assignment_args[max_p + i]
                    if dd_val == "Skip" or dd_val not in pool_by_label:
                        file_assignments.append(None)
                    else:
                        g = pool_by_label[dd_val]
                        g_parsed = parse_filename(Path(g["path"]).name)
                        garment_local = Path(g["path"])
                        if garment_local.exists() and str(garment_local).startswith(str(LOCAL_DATA)):
                            upload_image(garment_local, str(garment_local.relative_to(LOCAL_DATA)))
                        garment_id = g_parsed["id"] if g_parsed else generate_id()
                        file_assignments.append({"garment_id": garment_id, "category": cat_val or "tops"})
                save_multi_result(UPLOADS_PREFIX, p_parsed["id"], file_assignments, result)
        return result

    with gr.Blocks(title="Multi-Person Virtual Try-On") as demo:
        with gr.Tabs():
            # ---- Main VTON Tab ----
            with gr.Tab("Virtual Try-On"):
                gr.Markdown("# Multi-Person Virtual Try-On")
                gr.Markdown("Select a portrait, add garments to the pool, detect people, and assign garments.")

                selected_portrait = gr.State(value=None)
                garment_pool = gr.State(value=[])
                num_detected = gr.State(value=0)
                garment_counter = gr.State(value=0)

                # Portrait section
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Portrait")
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
                        preview_portrait = gr.Image(label="Selected Portrait", interactive=False, height=250)

                    # Garment pool section
                    with gr.Column():
                        gr.Markdown("### Garment Pool")
                        garment_gallery = gr.Gallery(
                            value=_gallery_images(UPLOADS_PREFIX, "garments"),
                            label="Available Garments (click to add to pool)",
                            columns=4,
                            height=200,
                            allow_preview=False,
                        )
                        with gr.Accordion("Upload new garment", open=False):
                            garment_upload = gr.Image(type="pil", label="Upload Garment", sources=["upload", "webcam"])
                            with gr.Row():
                                garment_url_input = gr.Textbox(label="Or paste image URL", scale=4)
                                garment_url_btn = gr.Button("Load", size="sm", scale=1)
                        garment_pool_gallery = gr.Gallery(
                            label="Current Pool",
                            columns=6,
                            height=120,
                            allow_preview=False,
                        )
                        clear_pool_btn = gr.Button("Clear Pool", size="sm", variant="stop")

                # Detection section
                detect_btn = gr.Button("Detect People", variant="secondary")
                detect_status = gr.Textbox(interactive=False, show_label=False, value="")
                people_gallery = gr.Gallery(
                    label="Detected People",
                    columns=6,
                    height=300,
                    allow_preview=False,
                )

                # Per-person assignment panel
                gr.Markdown("### Assign Garments to People")
                assignment_dropdowns = []
                assignment_categories = []
                assignment_rows = []

                for i in range(max_people):
                    with gr.Row(visible=False) as row:
                        dd = gr.Dropdown(
                            choices=["Skip"],
                            value="Skip",
                            label=f"Person {i + 1} — Garment",
                            scale=3,
                        )
                        cat = gr.Radio(
                            choices=["tops", "bottoms", "one-pieces"],
                            value="tops",
                            label="Category",
                            scale=2,
                        )
                        assignment_dropdowns.append(dd)
                        assignment_categories.append(cat)
                        assignment_rows.append(row)

                submit_btn = gr.Button("Try On", variant="primary")
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

                def on_portrait_gallery_select(evt: gr.SelectData):
                    path = evt.value["image"]["path"]
                    local_path = download_to_local(path)
                    return local_path, local_path

                def on_garment_gallery_select(evt: gr.SelectData, pool, counter):
                    """Add selected garment to pool."""
                    path = evt.value["image"]["path"]
                    local_path = download_to_local(path)
                    new_counter = counter + 1
                    label = f"Garment {new_counter}"
                    new_pool = pool + [{"path": local_path, "label": label}]
                    pool_images = [g["path"] for g in new_pool]
                    choices = ["Skip"] + [g["label"] for g in new_pool]
                    dd_updates = [gr.update(choices=choices) for _ in range(max_people)]
                    return [new_pool, new_counter, pool_images] + dd_updates

                def clear_pool():
                    dd_updates = [gr.update(choices=["Skip"], value="Skip") for _ in range(max_people)]
                    return [[], 0, [], _gallery_images(UPLOADS_PREFIX, "garments")] + dd_updates

                def reset_detection():
                    row_updates = [gr.Row(visible=False) for _ in range(max_people)]
                    dd_updates = [gr.update(value="Skip") for _ in range(max_people)]
                    cat_updates = [gr.update(value="tops") for _ in range(max_people)]
                    return ["", [], 0] + row_updates + dd_updates + cat_updates

                detection_reset_outputs = (
                    [detect_status, people_gallery, num_detected]
                    + assignment_rows
                    + assignment_dropdowns
                    + assignment_categories
                )

                portrait_gallery.select(
                    on_portrait_gallery_select, outputs=[selected_portrait, preview_portrait]
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_gallery.select(
                    on_garment_gallery_select,
                    inputs=[garment_pool, garment_counter],
                    outputs=[garment_pool, garment_counter, garment_pool_gallery] + assignment_dropdowns,
                )

                ex_portrait_gallery.select(
                    on_portrait_gallery_select, outputs=[selected_portrait, preview_portrait]
                ).then(reset_detection, outputs=detection_reset_outputs)
                ex_garment_gallery.select(
                    on_garment_gallery_select,
                    inputs=[garment_pool, garment_counter],
                    outputs=[garment_pool, garment_counter, garment_pool_gallery] + assignment_dropdowns,
                )

                clear_pool_btn.click(
                    clear_pool,
                    outputs=[garment_pool, garment_counter, garment_pool_gallery, garment_gallery] + assignment_dropdowns,
                )

                def on_portrait_upload(img, current_pool):
                    if img is None:
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, None
                    item_id = generate_id()
                    cat = "tops"  # default category for portrait filename
                    fname = make_filename(item_id, cat, "portrait")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), path, path, None

                def on_garment_upload(img, pool, counter):
                    if img is None:
                        choices = ["Skip"] + [g["label"] for g in pool]
                        dd_updates = [gr.update(choices=choices) for _ in range(max_people)]
                        return [_gallery_images(UPLOADS_PREFIX, "garments"), pool, counter, [g["path"] for g in pool], None] + dd_updates
                    item_id = generate_id()
                    cat = "tops"
                    fname = make_filename(item_id, cat, "garment")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    new_counter = counter + 1
                    label = f"Garment {new_counter}"
                    new_pool = pool + [{"path": path, "label": label}]
                    pool_images = [g["path"] for g in new_pool]
                    choices = ["Skip"] + [g["label"] for g in new_pool]
                    dd_updates = [gr.update(choices=choices) for _ in range(max_people)]
                    return [_gallery_images(UPLOADS_PREFIX, "garments"), new_pool, new_counter, pool_images, None] + dd_updates

                def on_portrait_url(url, pool):
                    if not url or not url.strip():
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, ""
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, "tops", "portrait")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), local_path, local_path, ""

                def on_garment_url(url, pool, counter):
                    if not url or not url.strip():
                        choices = ["Skip"] + [g["label"] for g in pool]
                        dd_updates = [gr.update(choices=choices) for _ in range(max_people)]
                        return [_gallery_images(UPLOADS_PREFIX, "garments"), pool, counter, [g["path"] for g in pool], ""] + dd_updates
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, "tops", "garment")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    new_counter = counter + 1
                    label = f"Garment {new_counter}"
                    new_pool = pool + [{"path": local_path, "label": label}]
                    pool_images = [g["path"] for g in new_pool]
                    choices = ["Skip"] + [g["label"] for g in new_pool]
                    dd_updates = [gr.update(choices=choices) for _ in range(max_people)]
                    return [_gallery_images(UPLOADS_PREFIX, "garments"), new_pool, new_counter, pool_images, ""] + dd_updates

                portrait_upload.change(
                    on_portrait_upload,
                    inputs=[portrait_upload, garment_pool],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_upload],
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_upload.change(
                    on_garment_upload,
                    inputs=[garment_upload, garment_pool, garment_counter],
                    outputs=[garment_gallery, garment_pool, garment_counter, garment_pool_gallery, garment_upload] + assignment_dropdowns,
                )

                portrait_url_btn.click(
                    on_portrait_url,
                    inputs=[portrait_url_input, garment_pool],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_url_input],
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_url_btn.click(
                    on_garment_url,
                    inputs=[garment_url_input, garment_pool, garment_counter],
                    outputs=[garment_gallery, garment_pool, garment_counter, garment_pool_gallery, garment_url_input] + assignment_dropdowns,
                )

                def on_detect(portrait_path, pool):
                    if detect_fn is None or portrait_path is None:
                        raise gr.Error("Please select a portrait first.")
                    people = detect_fn(portrait_path)
                    n = len(people)
                    choices = ["Skip"] + [g["label"] for g in pool]
                    default_garment = choices[1] if len(choices) > 1 else "Skip"
                    row_updates = []
                    dd_updates = []
                    cat_updates = []
                    for i in range(max_people):
                        if i < n:
                            row_updates.append(gr.Row(visible=True))
                            dd_updates.append(gr.update(choices=choices, value=default_garment))
                            cat_updates.append(gr.update(value="tops"))
                        else:
                            row_updates.append(gr.Row(visible=False))
                            dd_updates.append(gr.update(choices=choices, value="Skip"))
                            cat_updates.append(gr.update(value="tops"))
                    return (
                        [f"Found {n} {'person' if n == 1 else 'people'}", people, n]
                        + row_updates + dd_updates + cat_updates
                    )

                detect_btn.click(
                    lambda: "Detecting people...",
                    outputs=[detect_status],
                ).then(
                    on_detect,
                    inputs=[selected_portrait, garment_pool],
                    outputs=[detect_status, people_gallery, num_detected]
                    + assignment_rows + assignment_dropdowns + assignment_categories,
                )

                submit_btn.click(
                    process_and_save,
                    inputs=[selected_portrait, garment_pool, num_detected]
                    + assignment_dropdowns + assignment_categories,
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
                    fname = Path(result_local).name
                    # Try old single-garment format first
                    parsed = parse_result_filename(fname)
                    if not parsed:
                        parsed = parse_result_filename(Path(path).name)
                    if parsed:
                        cat = parsed["category"]
                        try:
                            portrait_url = file_url(f"{UPLOADS_PREFIX}/portraits/{make_filename(parsed['portrait_id'], cat, 'portrait')}")
                            garment_url = file_url(f"{UPLOADS_PREFIX}/garments/{make_filename(parsed['garment_id'], cat, 'garment')}")
                            portrait_local = download_to_local(portrait_url)
                            garment_local = download_to_local(garment_url)
                            return portrait_local, garment_local, result_local, cat, portrait_local, garment_local, result_local
                        except Exception:
                            pass
                    # Try multi-garment format
                    multi = parse_multi_result_filename(fname)
                    if not multi:
                        multi = parse_multi_result_filename(Path(path).name)
                    if multi:
                        # Find first non-None assignment for promote
                        first_assignment = next((a for a in multi["assignments"] if a is not None), None)
                        if first_assignment:
                            cat = first_assignment["category"]
                            try:
                                portrait_url = file_url(f"{UPLOADS_PREFIX}/portraits/{make_filename(multi['portrait_id'], cat, 'portrait')}")
                                garment_url = file_url(f"{UPLOADS_PREFIX}/garments/{make_filename(first_assignment['garment_id'], cat, 'garment')}")
                                portrait_local = download_to_local(portrait_url)
                                garment_local = download_to_local(garment_url)
                                return portrait_local, garment_local, result_local, cat, portrait_local, garment_local, result_local
                            except Exception:
                                pass
                    gr.Warning("Could not find matching portrait/garment.")
                    return None, None, result_local, "tops", None, None, result_local

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
    def dummy_process(portrait, pool, num_detected, *assignment_args):
        return Image.new("RGB", (512, 512), (200, 200, 200))
    def dummy_detect(portrait_path):
        return [Image.new("RGB", (100, 200), (255, 0, 0)), Image.new("RGB", (100, 200), (0, 255, 0))]
    demo = build_demo(dummy_process, detect_fn=dummy_detect)
    demo.launch()
