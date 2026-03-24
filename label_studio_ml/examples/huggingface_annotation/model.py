import os
import io
import logging
import requests
import mimetypes

from typing import List, Dict, Optional
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

HUGGINGFACE_API_BASE = "https://router.huggingface.co/hf-inference/models"


class HuggingFaceAnnotationModel(LabelStudioMLBase):
    """
    Generic Label Studio ML backend that connects to any Hugging Face model
    via the Inference API.

    Supported task types (auto-detected from the LS label config):
      - Object Detection   → RectangleLabels + Image
      - NER                → Labels + Text
      - Text Classification → Choices + Text
      - Image Classification → Choices + Image

    Required environment variables:
      HUGGINGFACE_API_KEY   – Your HF Inference API token
      MODEL_NAME            – e.g. facebook/detr-resnet-50 or dslim/bert-base-NER
      LABEL_STUDIO_URL      – e.g. http://localhost:8080
      LABEL_STUDIO_API_KEY  – Your Label Studio API key
    """

    def setup(self):
        model_name = os.getenv("MODEL_NAME", "unknown")
        self.set("model_version", f"{model_name}-v1")
        logger.info(f"HuggingFaceAnnotationModel set up with model: {model_name}")

    # ------------------------------------------------------------------
    # Hugging Face API helpers
    # ------------------------------------------------------------------

    def _get_api_url(self) -> Optional[str]:
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            logger.error("MODEL_NAME environment variable is not set")
            return None
        return f"{HUGGINGFACE_API_BASE}/{model_name}"

    def _get_hf_headers(self) -> Optional[Dict]:
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            logger.error("HUGGINGFACE_API_KEY environment variable is not set")
            return None
        return {"Authorization": f"Bearer {api_key}"}

    def _call_hf_api(self, *, json_data=None, binary_data=None, mime_type="application/octet-stream"):
        """
        Call the Hugging Face Inference API.
        Provide either json_data (dict) for text tasks or binary_data (bytes) for image tasks.
        Returns the parsed JSON response or None on failure.
        """
        url = self._get_api_url()
        if not url:
            return None

        headers = self._get_hf_headers()
        if not headers:
            return None

        try:
            if binary_data is not None:
                # Add content type header strictly required by router.huggingface.co
                headers["Content-Type"] = mime_type
                response = requests.post(url, headers=headers, data=binary_data, timeout=60)
            elif json_data is not None:
                # Automatically sets 'Content-Type: application/json'
                response = requests.post(url, headers=headers, json=json_data, timeout=60)
            else:
                logger.error("Both json_data and binary_data are None. Cannot call Hugging Face API.")
                return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error("Hugging Face API request timed out")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Hugging Face API HTTP error: {e.response.status_code} – {e.response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling Hugging Face API: {e}")

        return None

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        for task in tasks:
            try:
                pred = self._predict_single_task(task)
            except Exception as e:
                logger.error(f"Error predicting task {task.get('id')}: {e}", exc_info=True)
                pred = {"result": [], "score": 0.0, "model_version": self.get("model_version")}
            predictions.append(pred)

        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))

    # ------------------------------------------------------------------
    # Task-type routing
    # ------------------------------------------------------------------

    def _predict_single_task(self, task: Dict) -> Dict:
        li = self.label_interface

        # 1. Object Detection
        match = li.get_first_tag_occurence("RectangleLabels", "Image")
        if match:
            from_name, to_name, value_key = match
            return self._predict_object_detection(task, from_name, to_name, value_key)

        # 2. Named Entity Recognition
        match = li.get_first_tag_occurence("Labels", "Text")
        if match:
            from_name, to_name, value_key = match
            return self._predict_ner(task, from_name, to_name, value_key)

        # 3. Text Classification
        match = li.get_first_tag_occurence("Choices", "Text")
        if match:
            from_name, to_name, value_key = match
            return self._predict_text_classification(task, from_name, to_name, value_key)

        # 4. Image Classification
        match = li.get_first_tag_occurence("Choices", "Image")
        if match:
            from_name, to_name, value_key = match
            return self._predict_image_classification(task, from_name, to_name, value_key)

        logger.warning(
            f"Task {task.get('id')}: No supported label type found in the label config. "
            "Supported: RectangleLabels+Image, Labels+Text, Choices+Text, Choices+Image"
        )
        return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

    # ------------------------------------------------------------------
    # Object Detection  (e.g. facebook/detr-resnet-50)
    # ------------------------------------------------------------------

    def _predict_object_detection(self, task, from_name, to_name, value_key) -> Dict:
        image_url = task["data"].get(value_key)
        if not image_url:
            logger.warning(f"Task {task.get('id')}: no image found at data['{value_key}']")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        # Check local path validity
        image_path = self.get_local_path(image_url, task_id=task.get("id"))
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Task {task.get('id')}: Image path is invalid or does not exist: {image_path}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Check bytes payload
        if not image_bytes:
            logger.error(f"Task {task.get('id')}: Read 0 bytes from {image_path}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        img = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = img.size

        # Detect the correct MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  # Safe fallback

        # Pass payload safely with the correct content type
        hf_response = self._call_hf_api(binary_data=image_bytes, mime_type=mime_type)
        if not isinstance(hf_response, list):
            logger.warning(f"Task {task.get('id')}: unexpected object detection response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        # Collect all unique labels from HF response to build a mapping
        hf_labels = list({item.get("label") for item in hf_response if item.get("label")})
        # Map HF labels to Label Studio config labels (case-insensitively, etc.)
        label_map = self.build_label_map(from_name, hf_labels)
        
        logger.debug(f"HF Labels: {hf_labels}")
        logger.debug(f"Mapped Labels: {label_map}")

        results = []
        scores = []
        for item in hf_response:
            box = item.get("box")
            raw_label = item.get("label")
            score = item.get("score", 0.0)
            
            if not box or not raw_label:
                continue

            # Map the raw label to the exact label expected by Label Studio
            mapped_label = label_map.get(raw_label)
            if not mapped_label:
                logger.debug(f"Skipping prediction with unmapped label: {raw_label}")
                continue

            results.append({
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "value": {
                    "x": box["xmin"] * 100 / img_width,
                    "y": box["ymin"] * 100 / img_height,
                    "width": (box["xmax"] - box["xmin"]) * 100 / img_width,
                    "height": (box["ymax"] - box["ymin"]) * 100 / img_height,
                    "rectanglelabels": [mapped_label],
                },
                "score": score,
            })
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"result": results, "score": avg_score, "model_version": self.get("model_version")}

    # ------------------------------------------------------------------
    # Named Entity Recognition  (e.g. dslim/bert-base-NER)
    # ------------------------------------------------------------------

    def _predict_ner(self, task, from_name, to_name, value_key) -> Dict:
        text = task["data"].get(value_key, "")
        if not text:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        hf_response = self._call_hf_api(json_data={"inputs": text})
        if not isinstance(hf_response, list):
            logger.warning(f"Task {task.get('id')}: unexpected NER response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        hf_labels = list({(item.get("entity_group") or item.get("entity")) for item in hf_response if (item.get("entity_group") or item.get("entity"))})
        label_map = self.build_label_map(from_name, hf_labels)

        results = []
        scores = []
        for item in hf_response:
            raw_label = item.get("entity_group") or item.get("entity")
            start = item.get("start")
            end = item.get("end")
            score = item.get("score", 0.0)

            if raw_label is None or start is None or end is None:
                continue

            mapped_label = label_map.get(raw_label)
            if not mapped_label:
                continue

            results.append({
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [mapped_label],
                },
                "score": score,
            })
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"result": results, "score": avg_score, "model_version": self.get("model_version")}

    # ------------------------------------------------------------------
    # Text Classification  (e.g. distilbert-base-uncased-finetuned-sst-2-english)
    # ------------------------------------------------------------------

    def _predict_text_classification(self, task, from_name, to_name, value_key) -> Dict:
        text = task["data"].get(value_key, "")
        if not text:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        hf_response = self._call_hf_api(json_data={"inputs": text})

        # HF can return [[{label, score}, ...]] or [{label, score}, ...]
        if isinstance(hf_response, list) and hf_response and isinstance(hf_response[0], list):
            hf_response = hf_response[0]

        if not isinstance(hf_response, list) or not hf_response:
            logger.warning(f"Task {task.get('id')}: unexpected classification response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        hf_labels = list({item.get("label") for item in hf_response if item.get("label")})
        label_map = self.build_label_map(from_name, hf_labels)

        # Pick the highest-confidence label that we can map
        mapped_results = []
        for item in hf_response:
            raw_label = item.get("label", "")
            mapped_label = label_map.get(raw_label)
            if mapped_label:
                mapped_results.append((mapped_label, item.get("score", 0.0)))

        if not mapped_results:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        best_label, best_score = max(mapped_results, key=lambda x: x[1])

        return {
            "result": [{
                "from_name": from_name,
                "to_name": to_name,
                "type": "choices",
                "value": {"choices": [best_label]},
            }],
            "score": best_score,
            "model_version": self.get("model_version"),
        }

    # ------------------------------------------------------------------
    # Image Classification  (e.g. google/vit-base-patch16-224)
    # ------------------------------------------------------------------

    def _predict_image_classification(self, task, from_name, to_name, value_key) -> Dict:
        image_url = task["data"].get(value_key)
        if not image_url:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        image_path = self.get_local_path(image_url, task_id=task.get("id"))
        if not image_path or not os.path.exists(image_path):
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        hf_response = self._call_hf_api(binary_data=image_bytes, mime_type=mime_type)

        # HF image classification returns [{label, score}, ...]
        if not isinstance(hf_response, list) or not hf_response:
            logger.warning(f"Task {task.get('id')}: unexpected image classification response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        hf_labels = list({item.get("label") for item in hf_response if item.get("label")})
        label_map = self.build_label_map(from_name, hf_labels)

        mapped_results = []
        for item in hf_response:
            raw_label = item.get("label", "")
            mapped_label = label_map.get(raw_label)
            if mapped_label:
                mapped_results.append((mapped_label, item.get("score", 0.0)))

        if not mapped_results:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        best_label, best_score = max(mapped_results, key=lambda x: x[1])

        return {
            "result": [{
                "from_name": from_name,
                "to_name": to_name,
                "type": "choices",
                "value": {"choices": [best_label]},
            }],
            "score": best_score,
            "model_version": self.get("model_version"),
        }
