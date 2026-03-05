import os
import uuid
from pathlib import Path
from PIL import Image
import numpy as np

DATA_REPO = "aj406/vton-data"
REPO_TYPE = "dataset"
DATASET_HF_TOKEN = os.environ.get("DATASET_HF_TOKEN")
LOCAL_DATA = Path("data")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def is_remote():
    return DATASET_HF_TOKEN is not None


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _ensure_repo():
    if not is_remote():
        return
    _api().create_repo(repo_id=DATA_REPO, repo_type=REPO_TYPE, exist_ok=True, token=DATASET_HF_TOKEN)


def save_image(img, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(local_path, "JPEG", quality=85)


def upload_image(local_path, remote_path):
    if not is_remote():
        return
    _ensure_repo()
    _api().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=DATA_REPO,
        repo_type=REPO_TYPE,
        token=DATASET_HF_TOKEN,
    )


def delete_remote_file(remote_path):
    if not is_remote():
        return
    _api().delete_file(
        path_in_repo=remote_path,
        repo_id=DATA_REPO,
        repo_type=REPO_TYPE,
        token=DATASET_HF_TOKEN,
    )


def download_dir(remote_prefix):
    if not is_remote():
        return
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type=REPO_TYPE,
        allow_patterns=f"{remote_prefix}/**",
        local_dir=str(LOCAL_DATA),
        token=DATASET_HF_TOKEN,
    )


def generate_id():
    return uuid.uuid4().hex[:8]


def make_filename(item_id, category, item_type):
    return f"{item_id}_{category}_{item_type}.jpg"


def parse_filename(filename):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    return {"id": parts[0], "category": parts[1], "type": parts[2]}


def list_local_images(directory):
    d = Path(directory)
    if not d.exists():
        return []
    return sorted([str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_image_sets(prefix):
    """Scan {prefix}/portraits/ dir, parse filenames, return list of dicts with matched files."""
    local_prefix = LOCAL_DATA / prefix
    if is_remote():
        download_dir(prefix)
    portraits_dir = local_prefix / "portraits"
    if not portraits_dir.exists():
        return []
    sets = {}
    for p in portraits_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        parsed = parse_filename(p.name)
        if not parsed:
            continue
        item_id = parsed["id"]
        category = parsed["category"]
        key = f"{item_id}_{category}"
        sets[key] = {
            "id": item_id,
            "category": category,
            "portrait": str(p),
        }
    garments_dir = local_prefix / "garments"
    results_dir = local_prefix / "results"
    for key, entry in sets.items():
        garment = garments_dir / f"{key}_garment.jpg"
        result = results_dir / f"{key}_result.jpg"
        entry["garment"] = str(garment) if garment.exists() else None
        entry["result"] = str(result) if result.exists() else None
    return [v for v in sets.values() if v["garment"] is not None]


def save_image_set(prefix, img_portrait, img_garment, category, img_result=None):
    """Save a set of images (portrait + garment + optional result) with consistent naming."""
    item_id = generate_id()
    local_prefix = LOCAL_DATA / prefix

    portrait_name = make_filename(item_id, category, "portrait")
    garment_name = make_filename(item_id, category, "garment")

    portrait_path = local_prefix / "portraits" / portrait_name
    garment_path = local_prefix / "garments" / garment_name

    save_image(img_portrait, portrait_path)
    save_image(img_garment, garment_path)
    upload_image(portrait_path, f"{prefix}/portraits/{portrait_name}")
    upload_image(garment_path, f"{prefix}/garments/{garment_name}")

    result_path = None
    if img_result is not None:
        result_name = make_filename(item_id, category, "result")
        result_path = local_prefix / "results" / result_name
        save_image(img_result, result_path)
        upload_image(result_path, f"{prefix}/results/{result_name}")

    return item_id, str(portrait_path), str(garment_path), str(result_path) if result_path else None


def save_result(prefix, item_id, category, img_result):
    """Save a result image for an existing image set."""
    local_prefix = LOCAL_DATA / prefix
    result_name = make_filename(item_id, category, "result")
    result_path = local_prefix / "results" / result_name
    save_image(img_result, result_path)
    upload_image(result_path, f"{prefix}/results/{result_name}")
    return str(result_path)


def delete_image_set(prefix, item_id, category):
    """Delete all files for an image set."""
    key = f"{item_id}_{category}"
    local_prefix = LOCAL_DATA / prefix
    for subdir, item_type in [("portraits", "portrait"), ("garments", "garment"), ("results", "result")]:
        local_file = local_prefix / subdir / f"{key}_{item_type}.jpg"
        if local_file.exists():
            local_file.unlink()
        if is_remote():
            try:
                delete_remote_file(f"{prefix}/{subdir}/{key}_{item_type}.jpg")
            except Exception:
                pass
