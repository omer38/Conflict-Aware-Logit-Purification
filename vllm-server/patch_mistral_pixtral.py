"""Patch vLLM for Mistral Pixtral + transformers >= 5.10 compatibility."""

from __future__ import annotations

from pathlib import Path

import vllm

_FETCH_IMAGES = '''
    def fetch_images(self, image_url_or_urls):
        from transformers.image_utils import is_valid_image, load_image

        if isinstance(image_url_or_urls, (list, tuple)):
            return [self.fetch_images(x) for x in image_url_or_urls]
        if isinstance(image_url_or_urls, str):
            return load_image(image_url_or_urls)
        if is_valid_image(image_url_or_urls):
            return image_url_or_urls
        raise TypeError(
            "only a single or a list of entries is supported but got "
            f"type={type(image_url_or_urls)}"
        )

'''

_NEEDLE = """    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int, int]:"""


def _patch(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "def fetch_images" in text:
        return False
    if _NEEDLE not in text:
        return False
    path.write_text(text.replace(_NEEDLE, _FETCH_IMAGES + _NEEDLE), encoding="utf-8")
    return True


def main() -> None:
    root = Path(vllm.__file__).resolve().parent
    targets = [
        root / "transformers_utils/processors/pixtral.py",
        root / "model_executor/models/pixtral.py",
    ]
    patched = [p for p in targets if _patch(p)]
    if patched:
        print("Patched MistralCommonImageProcessor.fetch_images in:")
        for p in patched:
            print(f"  {p}")
    else:
        print("Mistral Pixtral patch already applied.")


if __name__ == "__main__":
    main()
