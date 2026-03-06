import os
import uuid
import shutil
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


def make_filename(item_id, item_type):
    return f"{item_id}_{item_type}.jpg"


def parse_filename(filename):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    return {"id": parts[0], "type": parts[1]}


def make_result_filename(portrait_id, garment_id):
    return f"{portrait_id}_{garment_id}_result.jpg"


def parse_result_filename(filename):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3 or parts[2] != "result":
        return None
    if "-" in parts[1]:
        return None
    return {"portrait_id": parts[0], "garment_id": parts[1]}


def make_multi_result_filename(portrait_id, garment_ids):
    """Build result filename encoding per-person garment assignments.

    garment_ids: list of garment_id (str) or None per person.
    Example: portrait_id=abc123, garment_ids=["ef12ab34", None, "gh56cd78"]
    → "abc123_ef12ab34-x-gh56cd78_result.jpg"
    """
    slots = [gid if gid else "x" for gid in garment_ids]
    code = "-".join(slots)
    return f"{portrait_id}_{code}_result.jpg"


def parse_multi_result_filename(filename):
    stem = Path(filename).stem
    if not stem.endswith("_result"):
        return None
    stem = stem[:-len("_result")]
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    portrait_id = parts[0]
    code = parts[1]
    slots = code.split("-")
    garment_ids = [None if slot == "x" else slot for slot in slots]
    return {"portrait_id": portrait_id, "garment_ids": garment_ids}


def list_local_images(directory):
    d = Path(directory)
    if not d.exists():
        return []
    return sorted([str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def file_url(remote_path):
    """Return a direct HF URL for a file in the dataset repo (public repo)."""
    return f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/{remote_path}"


def list_gallery_urls(prefix, subdir):
    """List files in dataset repo and return direct URLs for gallery display."""
    if not is_remote():
        return list_local_images(LOCAL_DATA / prefix / subdir)
    try:
        items = _api().list_repo_tree(
            DATA_REPO, repo_type=REPO_TYPE, path_in_repo=f"{prefix}/{subdir}"
        )
        urls = []
        for item in items:
            if hasattr(item, "rfilename"):
                name = item.rfilename
            elif hasattr(item, "path"):
                name = item.path
            else:
                continue
            if Path(name).suffix.lower() in IMG_EXTS:
                urls.append(file_url(name))
        return sorted(urls)
    except Exception:
        return list_local_images(LOCAL_DATA / prefix / subdir)


HF_URL_PREFIX = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/"


def is_dataset_url(url):
    """Check if a URL points to our HF dataset repo."""
    return isinstance(url, str) and url.startswith(HF_URL_PREFIX)


def download_to_local(path_or_url):
    """Download a URL to local path. HF dataset URLs use hf_hub, other URLs use requests."""
    if not isinstance(path_or_url, str):
        return path_or_url
    if is_dataset_url(path_or_url):
        remote_path = path_or_url[len(HF_URL_PREFIX):]
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=DATA_REPO,
            repo_type=REPO_TYPE,
            filename=remote_path,
            token=DATASET_HF_TOKEN,
        )
        return local
    if path_or_url.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        tmp_path = LOCAL_DATA / "tmp" / f"{generate_id()}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(img, tmp_path)
        return str(tmp_path)
    return path_or_url


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
        sets[item_id] = {
            "id": item_id,
            "portrait": str(p),
        }
    garments_dir = local_prefix / "garments"
    results_dir = local_prefix / "results"
    for item_id, entry in sets.items():
        garment = garments_dir / f"{item_id}_garment.jpg"
        result = results_dir / f"{item_id}_result.jpg"
        entry["garment"] = str(garment) if garment.exists() else None
        entry["result"] = str(result) if result.exists() else None
    return [v for v in sets.values() if v["garment"] is not None]


def save_image_set(prefix, img_portrait, img_garment, img_result=None):
    """Save a set of images (portrait + garment + optional result) with consistent naming."""
    item_id = generate_id()
    local_prefix = LOCAL_DATA / prefix

    portrait_name = make_filename(item_id, "portrait")
    garment_name = make_filename(item_id, "garment")

    portrait_path = local_prefix / "portraits" / portrait_name
    garment_path = local_prefix / "garments" / garment_name

    save_image(img_portrait, portrait_path)
    save_image(img_garment, garment_path)
    upload_image(portrait_path, f"{prefix}/portraits/{portrait_name}")
    upload_image(garment_path, f"{prefix}/garments/{garment_name}")

    result_path = None
    if img_result is not None:
        result_name = make_result_filename(item_id, item_id)
        result_path = local_prefix / "results" / result_name
        save_image(img_result, result_path)
        upload_image(result_path, f"{prefix}/results/{result_name}")

    return item_id, str(portrait_path), str(garment_path), str(result_path) if result_path else None


def save_result(prefix, portrait_id, garment_id, img_result):
    """Save a result image encoding both portrait and garment IDs."""
    local_prefix = LOCAL_DATA / prefix
    result_name = make_result_filename(portrait_id, garment_id)
    result_path = local_prefix / "results" / result_name
    save_image(img_result, result_path)
    upload_image(result_path, f"{prefix}/results/{result_name}")
    return str(result_path)


def save_multi_result(prefix, portrait_id, assignments, img_result):
    """Save a multi-garment result image with assignment-encoded filename."""
    local_prefix = LOCAL_DATA / prefix
    result_name = make_multi_result_filename(portrait_id, assignments)
    result_path = local_prefix / "results" / result_name
    save_image(img_result, result_path)
    upload_image(result_path, f"{prefix}/results/{result_name}")
    return str(result_path)


def delete_image_set(prefix, item_id):
    """Delete all files for an image set (scans for ID prefix to catch multi-garment files)."""
    local_prefix = LOCAL_DATA / prefix
    for subdir in ("portraits", "garments", "results"):
        d = local_prefix / subdir
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.stem.startswith(item_id):
                f.unlink()
                if is_remote():
                    try:
                        delete_remote_file(f"{prefix}/{subdir}/{f.name}")
                    except Exception:
                        pass


def promote_to_example(result_path):
    """Copy a result file to examples, preserving its filename for resolution."""
    src = Path(result_path)
    dest = LOCAL_DATA / "examples" / "results" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))
    upload_image(dest, f"examples/results/{src.name}")
    return src.stem
