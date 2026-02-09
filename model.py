from fashn_vton import TryOnPipeline
from PIL import Image


if __name__ == "__main__":
    device = "cuda"
    pipeline = TryOnPipeline(weights_dir="./weights", device=device)

    person = Image.open("fashn-vton-1.5/examples/data/model.webp").convert("RGB")
    garment = Image.open("fashn-vton-1.5/examples/data/garment.webp").convert("RGB")

    # Run inference
    result = pipeline(
        person_image=person,
        garment_image=garment,
        category="tops",  # "tops" | "bottoms" | "one-pieces"
    )

    # Save output
    result.images[0].save("output.png")
