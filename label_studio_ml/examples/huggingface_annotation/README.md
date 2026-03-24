# Hugging Face Annotation ML Backend for Label Studio

A **generic, production-ready** Label Studio ML backend that connects any
[Hugging Face Inference API](https://huggingface.co/inference-api) model to Label Studio.
It automatically detects the annotation task type from your Label Studio label configuration
and returns predictions in the correct format.

---

## Supported Task Types

| Label Studio Setup | Example HF Models |
|---|---|
| **Object Detection** (`RectangleLabels` + `Image`) | `facebook/detr-resnet-50`, `hustvl/yolos-tiny` |
| **NER** (`Labels` + `Text`) | `dslim/bert-base-NER`, `Jean-Baptiste/roberta-large-ner-english` |
| **Text Classification** (`Choices` + `Text`) | `distilbert-base-uncased-finetuned-sst-2-english`, `cardiffnlp/twitter-roberta-base-sentiment` |
| **Image Classification** (`Choices` + `Image`) | `google/vit-base-patch16-224`, `microsoft/resnet-50` |

---

## Quick Start

### 1. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `HUGGINGFACE_API_KEY` | ✅ | Your Hugging Face API token ([get it here](https://huggingface.co/settings/tokens)) |
| `MODEL_NAME` | ✅ | HF model ID, e.g. `facebook/detr-resnet-50` |
| `LABEL_STUDIO_URL` | ✅ | Label Studio URL, e.g. `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | ✅ | Label Studio API key (Account → Access Token) |
| `LOG_LEVEL` | ❌ | Logging verbosity: `DEBUG`, `INFO` (default), `WARNING`, `ERROR` |
| `PORT` | ❌ | Server port (default: `9090`) |
| `WORKERS` | ❌ | Gunicorn workers (default: `1`) |
| `THREADS` | ❌ | Gunicorn threads (default: `8`) |
| `BASIC_AUTH_USER` | ❌ | Enable basic auth (optional) |
| `BASIC_AUTH_PASS` | ❌ | Enable basic auth (optional) |

### 2. Start the Backend

```bash
docker compose up --build
```

The backend will be available at `http://localhost:9090`.

### 3. Connect to Label Studio

1. Open your Label Studio project → **Settings** → **Model**
2. Add ML Backend with URL: `http://localhost:9090`
3. Enable **Auto-Annotation** in the project settings

---

## Example Models by Task

### Object Detection

```env
MODEL_NAME=facebook/detr-resnet-50
```

**Label Config:**
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="cat"/>
    <Label value="dog"/>
    <Label value="person"/>
  </RectangleLabels>
</View>
```

---

### Named Entity Recognition (NER)

```env
MODEL_NAME=dslim/bert-base-NER
```

**Label Config:**
```xml
<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="PER"/>
    <Label value="ORG"/>
    <Label value="LOC"/>
    <Label value="MISC"/>
  </Labels>
</View>
```

---

### Text Classification

```env
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
```

**Label Config:**
```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text">
    <Choice value="POSITIVE"/>
    <Choice value="NEGATIVE"/>
  </Choices>
</View>
```

---

### Image Classification

```env
MODEL_NAME=google/vit-base-patch16-224
```

**Label Config:**
```xml
<View>
  <Image name="image" value="$image"/>
  <Choices name="label" toName="image">
    <Choice value="cat"/>
    <Choice value="dog"/>
  </Choices>
</View>
```

---

## Project Structure

```
huggingface_annotation/
├── model.py            # Core ML model (auto-detects task type)
├── _wsgi.py            # WSGI entry point (gunicorn-compatible)
├── requirements.txt    # Example-specific Python dependencies
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Docker Compose configuration
├── .env.example        # Environment variable template
└── README.md           # This file
```

## How It Works

1. **Task Detection**: On the first prediction request, the model reads the Label Studio label configuration and detects the annotation task type (object detection, NER, text classification, or image classification).
2. **HF API Call**: It calls `https://api-inference.huggingface.co/models/{MODEL_NAME}` with the appropriate input format for the detected task.
3. **Response Mapping**: The raw HF API response is mapped to the Label Studio prediction JSON schema and returned.

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: label_studio_ml.response` | Build with the repo root as context (see Dockerfile) |
| `401 Unauthorized` on HF API | Check `HUGGINGFACE_API_KEY` is set and valid |
| `503 Service Unavailable` | Model may be loading on HF free tier — retry after ~30 seconds |
| Predictions not showing | Check that the label names in your config match the model's output labels |
| Empty predictions | Set `LOG_LEVEL=DEBUG` and check container logs for API errors |
