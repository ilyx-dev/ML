from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI()

# Проверка наличия PyTorch
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    print("Warning: Running on CPU. Consider installing PyTorch with GPU support for better performance.")

# Загрузка модели для анализа настроений
try:
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Модель для валидации входных данных
class ReviewInput(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва для анализа")

@app.get("/")
def read_root():
    """Возвращает информацию об API."""
    return {"message": "API для анализа настроений отзывов о ресторанах"}

@app.post("/predict")
def predict(review: ReviewInput):
    """Предсказывает настроение отзыва (положительное/отрицательное)."""
    try:
        result = classifier(review.text)[0]
        return {"sentiment": result["label"], "confidence": result["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)