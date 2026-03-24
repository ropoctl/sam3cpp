from pathlib import Path
import os

from sam3cpp import Sam3Model, draw_overlay


def main():
    repo = Path(__file__).resolve().parents[3]
    prefer_gpu = os.environ.get("SAM3CPP_EXAMPLE_GPU", "").lower() in {"1", "true", "yes", "on"}
    model = Sam3Model(str(repo / "models/sam3-image-f32.gguf"), prefer_gpu=prefer_gpu)
    image = Path("/Users/rpo/Downloads/tokens.jpg")
    prediction = model.predict(str(image), "person")
    out = draw_overlay(str(image), prediction)
    out_path = repo / "out" / "bindings_python_tokens_person.png"
    out.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
