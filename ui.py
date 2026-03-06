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
                # Upload all garments in pool
                for g in garment_pool:
                    garment_local = Path(g["path"])
                    if garment_local.exists() and str(garment_local).startswith(str(LOCAL_DATA)):
                        upload_image(garment_local, str(garment_local.relative_to(LOCAL_DATA)))
                # Build garment ID list for filename
                n = num_detected if num_detected else 0
                max_p = len(assignment_args) // 2
                pool_by_label = {g["label"]: g for g in garment_pool}
                garment_ids = []
                for i in range(n):
                    dd_val = assignment_args[i]
                    if dd_val == "Skip" or dd_val not in pool_by_label:
                        garment_ids.append(None)
                    else:
                        g = pool_by_label[dd_val]
                        g_parsed = parse_filename(Path(g["path"]).name)
                        garment_ids.append(g_parsed["id"] if g_parsed else generate_id())
                save_multi_result(UPLOADS_PREFIX, p_parsed["id"], garment_ids, result)
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

                # Per-person assignment panel (dynamically rendered)
                @gr.render(inputs=[num_detected, garment_pool])
                def render_assignments(n_detected, pool):
                    n = n_detected or 0
                    choices = ["Skip"] + [g["label"] for g in (pool or [])]
                    default_garment = choices[1] if len(choices) > 1 else "Skip"

                    if n > 0:
                        gr.Markdown(f"### Assign Garments to {n} {'Person' if n == 1 else 'People'}")

                    dds = []
                    cats = []
                    for i in range(n):
                        with gr.Row():
                            dd = gr.Dropdown(
                                choices=choices,
                                value=default_garment,
                                label=f"Person {i + 1} — Garment",
                                scale=3,
                                interactive=True,
                            )
                            cat = gr.Radio(
                                choices=["tops", "bottoms", "one-pieces"],
                                value="tops",
                                label="Category",
                                scale=2,
                                interactive=True,
                            )
                            dds.append(dd)
                            cats.append(cat)

                    submit_btn = gr.Button("Try On", variant="primary")
                    result_image = gr.Image(type="pil", label="Result", interactive=False)

                    submit_btn.click(
                        process_and_save,
                        inputs=[selected_portrait, garment_pool, num_detected] + dds + cats,
                        outputs=result_image,
                    )

                # Examples section
                gr.Markdown("---")
                gr.Markdown("### Examples")
                example_sets = gr.State(value=[])
                refresh_examples_btn = gr.Button("Refresh Examples", size="sm")

                @gr.render(inputs=[example_sets])
                def render_examples(sets):
                    for i, ex in enumerate(sets or []):
                        with gr.Row():
                            gr.Image(value=ex["portrait"], label="Portrait", height=200, interactive=False, scale=1)
                            for j, g in enumerate(ex["garments"]):
                                gr.Image(value=g, label=f"Garment {j+1}", height=200, interactive=False, scale=1)
                            gr.Image(value=ex["result"], label="Result", height=200, interactive=False, scale=1)
                            use_btn = gr.Button("Use", size="sm", scale=0, min_width=60)
                            use_btn.click(
                                lambda p=ex["portrait"], gs=ex["garments"]: _load_example(p, gs),
                                outputs=[selected_portrait, preview_portrait, garment_pool, garment_counter, garment_pool_gallery],
                            )

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
                    return new_pool, new_counter, pool_images

                def _load_example(portrait_path, garment_paths):
                    pool = [{"path": g, "label": f"Garment {i+1}"} for i, g in enumerate(garment_paths)]
                    pool_images = [g["path"] for g in pool]
                    return portrait_path, portrait_path, pool, len(pool), pool_images

                def clear_pool():
                    return [], 0, [], _gallery_images(UPLOADS_PREFIX, "garments")

                def reset_detection():
                    return "", [], 0

                detection_reset_outputs = [detect_status, people_gallery, num_detected]

                portrait_gallery.select(
                    on_portrait_gallery_select, outputs=[selected_portrait, preview_portrait]
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_gallery.select(
                    on_garment_gallery_select,
                    inputs=[garment_pool, garment_counter],
                    outputs=[garment_pool, garment_counter, garment_pool_gallery],
                )

                clear_pool_btn.click(
                    clear_pool,
                    outputs=[garment_pool, garment_counter, garment_pool_gallery, garment_gallery],
                )

                def on_portrait_upload(img, current_pool):
                    if img is None:
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, None
                    item_id = generate_id()
                    fname = make_filename(item_id, "portrait")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), path, path, None

                def on_garment_upload(img, pool, counter):
                    if img is None:
                        return _gallery_images(UPLOADS_PREFIX, "garments"), pool, counter, [g["path"] for g in pool], None
                    item_id = generate_id()
                    fname = make_filename(item_id, "garment")
                    local_path = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                    save_image(img, local_path)
                    path = str(local_path)
                    new_counter = counter + 1
                    label = f"Garment {new_counter}"
                    new_pool = pool + [{"path": path, "label": label}]
                    pool_images = [g["path"] for g in new_pool]
                    return _gallery_images(UPLOADS_PREFIX, "garments"), new_pool, new_counter, pool_images, None

                def on_portrait_url(url, pool):
                    if not url or not url.strip():
                        return _gallery_images(UPLOADS_PREFIX, "portraits"), None, None, ""
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, "portrait")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "portraits" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    return _gallery_images(UPLOADS_PREFIX, "portraits"), local_path, local_path, ""

                def on_garment_url(url, pool, counter):
                    if not url or not url.strip():
                        return _gallery_images(UPLOADS_PREFIX, "garments"), pool, counter, [g["path"] for g in pool], ""
                    local_path = download_to_local(url.strip())
                    if not is_dataset_url(url.strip()):
                        from PIL import Image as PILImage
                        item_id = generate_id()
                        fname = make_filename(item_id, "garment")
                        dest = LOCAL_DATA / UPLOADS_PREFIX / "garments" / fname
                        save_image(PILImage.open(local_path), dest)
                        local_path = str(dest)
                    new_counter = counter + 1
                    label = f"Garment {new_counter}"
                    new_pool = pool + [{"path": local_path, "label": label}]
                    pool_images = [g["path"] for g in new_pool]
                    return _gallery_images(UPLOADS_PREFIX, "garments"), new_pool, new_counter, pool_images, ""

                portrait_upload.change(
                    on_portrait_upload,
                    inputs=[portrait_upload, garment_pool],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_upload],
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_upload.change(
                    on_garment_upload,
                    inputs=[garment_upload, garment_pool, garment_counter],
                    outputs=[garment_gallery, garment_pool, garment_counter, garment_pool_gallery, garment_upload],
                )

                portrait_url_btn.click(
                    on_portrait_url,
                    inputs=[portrait_url_input, garment_pool],
                    outputs=[portrait_gallery, selected_portrait, preview_portrait, portrait_url_input],
                ).then(reset_detection, outputs=detection_reset_outputs)

                garment_url_btn.click(
                    on_garment_url,
                    inputs=[garment_url_input, garment_pool, garment_counter],
                    outputs=[garment_gallery, garment_pool, garment_counter, garment_pool_gallery, garment_url_input],
                )

                def on_detect(portrait_path, pool):
                    if detect_fn is None or portrait_path is None:
                        raise gr.Error("Please select a portrait first.")
                    people = detect_fn(portrait_path)
                    n = len(people)
                    return f"Found {n} {'person' if n == 1 else 'people'}", people, n

                detect_btn.click(
                    lambda: "Detecting people...",
                    outputs=[detect_status],
                ).then(
                    on_detect,
                    inputs=[selected_portrait, garment_pool],
                    outputs=[detect_status, people_gallery, num_detected],
                )

                def refresh_examples():
                    result_urls = _gallery_images(EXAMPLES_PREFIX, "results")
                    sets = []
                    for r in result_urls:
                        portrait, garments, result = _resolve_result_images(EXAMPLES_PREFIX, r)
                        if portrait and garments:
                            sets.append({"portrait": portrait, "garments": garments, "result": result})
                    return sets

                refresh_examples_btn.click(
                    refresh_examples,
                    outputs=[example_sets],
                )
                demo.load(
                    refresh_examples,
                    outputs=[example_sets],
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

                admin_upload_btn = gr.Button("Add Example", variant="primary")
                admin_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Current Examples")
                admin_examples_table = gr.Dataframe(
                    headers=["ID", "Result Filename"],
                    label="Examples",
                    interactive=False,
                )

                with gr.Row():
                    delete_id = gr.Textbox(label="Example ID to delete", scale=3)
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)

                def get_examples_table():
                    results = list_gallery_urls(EXAMPLES_PREFIX, "results")
                    rows = []
                    for r in results:
                        fname = Path(r).name
                        parsed = parse_result_filename(fname) or parse_multi_result_filename(fname)
                        rid = parsed["portrait_id"] if parsed else Path(fname).stem
                        rows.append([rid, fname])
                    return rows

                def _resolve_image(img, url):
                    if img is not None:
                        return img
                    if url and url.strip():
                        local = download_to_local(url.strip())
                        return Image.open(local)
                    return None

                def on_admin_upload(portrait, garment, result, p_url, g_url, r_url):
                    portrait = _resolve_image(portrait, p_url)
                    garment = _resolve_image(garment, g_url)
                    result = _resolve_image(result, r_url)
                    if portrait is None or garment is None:
                        return "Please provide at least portrait and garment (file or URL).", get_examples_table()
                    save_image_set(EXAMPLES_PREFIX, portrait, garment, result)
                    return "Example added.", get_examples_table()

                def on_admin_delete(ex_id):
                    if not ex_id:
                        return "Please provide an ID.", get_examples_table()
                    delete_image_set(EXAMPLES_PREFIX, ex_id.strip())
                    return "Deleted.", get_examples_table()

                admin_upload_btn.click(
                    on_admin_upload,
                    inputs=[admin_portrait, admin_garment, admin_result,
                            admin_portrait_url, admin_garment_url, admin_result_url],
                    outputs=[admin_status, admin_examples_table],
                )
                delete_btn.click(
                    on_admin_delete,
                    inputs=[delete_id],
                    outputs=[admin_status, admin_examples_table],
                )

                # ---- Promote from Uploads ----
                gr.Markdown("---")
                gr.Markdown("## Promote from Uploads")
                gr.Markdown("Select a result to promote. The matching portrait and garment are found automatically.")

                promote_portrait = gr.State(value=None)
                promote_garments = gr.State(value=[])
                promote_result = gr.State(value=None)

                promo_result_gallery = gr.Gallery(
                    value=_gallery_images(UPLOADS_PREFIX, "results"),
                    label="Results",
                    columns=4,
                    height=200,
                    allow_preview=False,
                )

                with gr.Row():
                    promo_preview_portrait = gr.Image(label="Portrait", interactive=False, height=150)
                    promo_preview_garments = gr.Gallery(label="Garments", columns=4, height=150, allow_preview=False)
                    promo_preview_result = gr.Image(label="Result", interactive=False, height=150)

                promote_btn = gr.Button("Promote to Example", variant="primary")
                promote_status = gr.Textbox(label="Status", interactive=False)

                def _resolve_result_images(prefix, path):
                    """Parse a result filename and resolve portrait + all garments."""
                    result_local = download_to_local(path)
                    fname = Path(result_local).name
                    # Try single-garment format
                    parsed = parse_result_filename(fname)
                    if not parsed:
                        parsed = parse_result_filename(Path(path).name)
                    if parsed:
                        try:
                            p_url = file_url(f"{prefix}/portraits/{make_filename(parsed['portrait_id'], 'portrait')}")
                            g_url = file_url(f"{prefix}/garments/{make_filename(parsed['garment_id'], 'garment')}")
                            return download_to_local(p_url), [download_to_local(g_url)], result_local
                        except Exception as e:
                            gr.Warning(f"Single-garment resolve failed for {fname}: {e}")
                    # Try multi-garment format
                    multi = parse_multi_result_filename(fname)
                    if not multi:
                        multi = parse_multi_result_filename(Path(path).name)
                    if multi:
                        gids = [gid for gid in multi["garment_ids"] if gid is not None]
                        if gids:
                            try:
                                p_url = file_url(f"{prefix}/portraits/{make_filename(multi['portrait_id'], 'portrait')}")
                                portrait_local = download_to_local(p_url)
                                garment_locals = []
                                for gid in gids:
                                    g_url = file_url(f"{prefix}/garments/{make_filename(gid, 'garment')}")
                                    garment_locals.append(download_to_local(g_url))
                                return portrait_local, garment_locals, result_local
                            except Exception as e:
                                gr.Warning(f"Multi-garment resolve failed for {fname} (portrait={multi['portrait_id']}, garments={gids}): {e}")
                    else:
                        gr.Warning(f"Could not parse result filename: {fname}")
                    return None, [], result_local

                def on_result_select(evt: gr.SelectData):
                    path = evt.value["image"]["path"]
                    portrait_local, garment_locals, result_local = _resolve_result_images(UPLOADS_PREFIX, path)
                    if not portrait_local or not garment_locals:
                        gr.Warning(f"Could not find matching portrait/garment for: {Path(path).name}")
                    return portrait_local, garment_locals, result_local, portrait_local, garment_locals, result_local

                promo_result_gallery.select(
                    on_result_select,
                    outputs=[promote_portrait, promote_garments, promote_result,
                             promo_preview_portrait, promo_preview_garments, promo_preview_result],
                )

                def on_promote(portrait_path, garment_paths, result_path):
                    if not portrait_path or not garment_paths:
                        return "Could not find matching portrait/garment.", get_examples_table()
                    item_id = promote_to_example(portrait_path, garment_paths, result_path)
                    return f"Promoted as example {item_id} ({len(garment_paths)} garment(s)).", get_examples_table()

                promote_btn.click(
                    on_promote,
                    inputs=[promote_portrait, promote_garments, promote_result],
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
