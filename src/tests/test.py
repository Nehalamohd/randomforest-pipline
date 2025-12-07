import pytest
from app import app
@pytest.fixture
def client():
    app.testing=True
    with app.test_client() as client:
        yield client

def test_app(client):
    data={"feature":[5.1, 3.5, 1.4, 0.2]}
    response=client.post('/predict',json=data)
    assert response.status_code==200
    json_data=response.get_json()
    assert "predict" in json_data
    assert 0<=json_data['predict']<=2