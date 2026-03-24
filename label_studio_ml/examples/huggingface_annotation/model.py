import os
import io
import logging
import requests

from typing import List, Dict, Optional
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

HUGGINGFACE_API_BASE = "https://api-inference.huggingface.co/models"


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

    def _call_hf_api(self, *, json_data=None, binary_data=None):
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
                response = requests.post(url, headers=headers, data=binary_data, timeout=30)
            else:
                response = requests.post(url, headers=headers, json=json_data, timeout=30)

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

        image_path = self.get_local_path(image_url, task_id=task.get("id"))
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        img = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = img.size

        hf_response = self._call_hf_api(binary_data=image_bytes)
        if not isinstance(hf_response, list):
            logger.warning(f"Task {task.get('id')}: unexpected object detection response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        results = []
        scores = []
        for item in hf_response:
            box = item.get("box")
            label = item.get("label")
            score = item.get("score", 0.0)
            if not box or not label:
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
                    "rectanglelabels": [label],
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

        results = []
        scores = []
        for item in hf_response:
            label = item.get("entity_group") or item.get("entity")
            start = item.get("start")
            end = item.get("end")
            score = item.get("score", 0.0)

            if label is None or start is None or end is None:
                continue

            results.append({
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [label],
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

        # Pick the highest-confidence label
        best = max(hf_response, key=lambda x: x.get("score", 0.0))
        label = best.get("label", "")
        score = best.get("score", 0.0)

        return {
            "result": [{
                "from_name": from_name,
                "to_name": to_name,
                "type": "choices",
                "value": {"choices": [label]},
            }],
            "score": score,
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
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        hf_response = self._call_hf_api(binary_data=image_bytes)

        # HF image classification returns [{label, score}, ...]
        if not isinstance(hf_response, list) or not hf_response:
            logger.warning(f"Task {task.get('id')}: unexpected image classification response: {hf_response}")
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        best = max(hf_response, key=lambda x: x.get("score", 0.0))
        label = best.get("label", "")
        score = best.get("score", 0.0)

        return {
            "result": [{
                "from_name": from_name,
                "to_name": to_name,
                "type": "choices",
                "value": {"choices": [label]},
            }],
            "score": score,
            "model_version": self.get("model_version"),
        }
