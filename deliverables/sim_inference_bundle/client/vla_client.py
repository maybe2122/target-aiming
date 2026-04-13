"""Standalone client for the EyeVLA inference service."""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class VLAClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 15.0

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, endpoint: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url.rstrip('/')}{endpoint}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _encode_image_bytes(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _pil_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def predict_bytes(
        self,
        image_bytes: bytes,
        instruction: str,
        predict_type: str = "grounding_action",
    ) -> dict[str, Any]:
        return self._post_json(
            "/predict",
            {
                "image": self._encode_image_bytes(image_bytes),
                "instruction": instruction,
                "type": predict_type,
            },
        )

    def predict_path(
        self,
        image_path: str | Path,
        instruction: str,
        predict_type: str = "grounding_action",
    ) -> dict[str, Any]:
        return self.predict_bytes(Path(image_path).read_bytes(), instruction, predict_type=predict_type)

    def predict_pil(
        self,
        image: Image.Image,
        instruction: str,
        predict_type: str = "grounding_action",
        quality: int = 95,
    ) -> dict[str, Any]:
        return self.predict_bytes(
            self._pil_to_jpeg_bytes(image, quality=quality),
            instruction,
            predict_type=predict_type,
        )

    def predict_cv2(
        self,
        cv_image,
        instruction: str,
        predict_type: str = "grounding_action",
        jpeg_quality: int = 95,
    ) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("predict_cv2 requires opencv-python to be installed") from exc

        ok, buf = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg', ...) failed")
        return self.predict_bytes(buf.tobytes(), instruction, predict_type=predict_type)

    def debug_grounding_path(self, image_path: str | Path, instruction: str) -> dict[str, Any]:
        return self._post_json(
            "/debug/grounding",
            {
                "image": self._encode_image_bytes(Path(image_path).read_bytes()),
                "instruction": instruction,
                "type": "grounding_only",
            },
        )


def main():
    parser = argparse.ArgumentParser(description="EyeVLA inference client")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--type",
        default="grounding_action",
        choices=["grounding_action", "action_only", "aligned_done", "no_target", "grounding_only"],
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    client = VLAClient(base_url=args.url, timeout=args.timeout)
    if args.type == "grounding_only":
        result = client.debug_grounding_path(args.image, args.instruction)
    else:
        result = client.predict_path(args.image, args.instruction, predict_type=args.type)
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
