from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .config import AppConfig


class ExtractionProvider(Protocol):
    def extract(self, pdf_path: str, schema_fields: list[str]) -> dict[str, Any]: ...


@dataclass
class LocalVLMProvider:
    endpoint: str
    timeout_s: int = 120

    def extract(self, pdf_path: str, schema_fields: list[str]) -> dict[str, Any]:
        # Expected contract: the local service returns JSON with keys in schema_fields.
        # This demo intentionally keeps it simple; you can swap the service implementation.
        payload = {"pdf_path": pdf_path, "schema_fields": schema_fields}
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("LocalVLM response is not a JSON object")
        return data


@dataclass
class NanonetsProvider:
    api_key: str
    endpoint: str  # https://app.nanonets.com/api/v2/OCR/Model/<MODEL_ID>/LabelFile/
    timeout_s: int = 120
    low_confidence_threshold: float = 0.8  # flag needs_review below this score

    def extract(self, pdf_path: str, schema_fields: list[str]) -> dict[str, Any]:
        """
        Call the Nanonets OCR API v2 and map the response to the flat schema dict.

        Nanonets response shape:
          {
            "result": [
              {
                "message": "Success",
                "prediction": [
                  {"label": "field_name", "ocr_text": "value", "score": 0.95, ...},
                  ...
                ]
              }
            ]
          }

        Endpoint format:
          POST https://app.nanonets.com/api/v2/OCR/Model/<MODEL_ID>/LabelFile/
          Auth: HTTP Basic  (api_key : "")
          Body: multipart/form-data  file=<pdf bytes>
        """
        with open(pdf_path, "rb") as fh:
            r = requests.post(
                self.endpoint,
                files={"file": fh},
                auth=requests.auth.HTTPBasicAuth(self.api_key, ""),
                timeout=self.timeout_s,
            )
        r.raise_for_status()
        body = r.json()
        return self._parse_response(body, schema_fields)

    def _parse_response(self, body: dict[str, Any], schema_fields: list[str]) -> dict[str, Any]:
        """Convert Nanonets API response to a flat {field: value} dict."""
        results = body.get("result", [])
        if not results:
            return {f: "[BRAK]" for f in schema_fields if f != "needs_review"} | {"needs_review": True}

        # Collect all predictions across all pages/results
        all_predictions: list[dict[str, Any]] = []
        for page_result in results:
            all_predictions.extend(page_result.get("prediction", []))

        # Build field → value map; first non-empty prediction per label wins
        field_map: dict[str, str] = {}
        low_confidence_fields: list[str] = []
        for pred in all_predictions:
            label: str = pred.get("label", "")
            text: str = (pred.get("ocr_text") or "").strip()
            score: float = float(pred.get("score", 1.0))

            if label and label not in field_map:
                field_map[label] = text if text else "[BRAK]"
                if score < self.low_confidence_threshold:
                    low_confidence_fields.append(label)

        # Map to schema fields; missing fields → [BRAK]
        output: dict[str, Any] = {}
        for field in schema_fields:
            if field == "needs_review":
                continue
            output[field] = field_map.get(field, "[BRAK]")

        needs_review = bool(
            low_confidence_fields
            or any(v == "[NIECZYTELNE]" for v in output.values())
        )
        output["needs_review"] = needs_review
        return output


def build_provider(cfg: AppConfig) -> ExtractionProvider:
    if cfg.provider.type == "local_vlm":
        return LocalVLMProvider(endpoint=cfg.local_vlm.endpoint, timeout_s=cfg.local_vlm.timeout_s)

    if cfg.provider.type == "nanonets":
        if not cfg.nanonets.endpoint:
            raise ValueError("nanonets.endpoint must be set in config.yaml when provider.type=nanonets")
        api_key = os.environ.get(cfg.nanonets.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key env var: {cfg.nanonets.api_key_env}")
        return NanonetsProvider(
            api_key=api_key,
            endpoint=cfg.nanonets.endpoint,
            low_confidence_threshold=cfg.nanonets.low_confidence_threshold,
        )

    raise ValueError(f"Unknown provider.type: {cfg.provider.type}")

