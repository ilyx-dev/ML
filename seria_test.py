from fastapi.testclient import TestClient
from seria import app

client = TestClient(app)

def test_read_root():
    """Проверка корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API для анализа настроений отзывов о ресторанах"}

def test_predict_positive():
    """Проверка предсказания для положительного отзыва."""
    response = client.post("/predict", json={"text": "Вкусная еда и отличный сервис!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "POSITIVE"
    assert 0 <= data["confidence"] <= 1

def test_predict_negative():
    """Проверка предсказания для отрицательного отзыва."""
    response = client.post("/predict", json={"text": "Еда была невкусной, долго ждали."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "NEGATIVE"
    assert 0 <= data["confidence"] <= 1

def test_empty_input():
    """Проверка обработки пустого текста."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_non_string_input():
    """Проверка обработки нестрокового ввода."""
    response = client.post("/predict", json={"text": 123})
    assert response.status_code == 422