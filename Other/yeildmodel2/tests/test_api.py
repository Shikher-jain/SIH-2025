import sys
import os
import pytest
import io
from PIL import Image

# Ensure tests can import the application from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_ndvi_mask_endpoint(client):
    ndvi = [[0.1, 0.5], [0.7, 0.2]]
    resp = client.post('/ndvi-mask', json={'ndvi': ndvi})
    assert resp.status_code == 200
    assert resp.content_type == 'image/png'
    img = Image.open(io.BytesIO(resp.data))
    assert img.mode == 'RGBA'


def test_bands_mode_endpoint(client):
    nir = [[0.6, 0.7], [0.8, 0.4]]
    red = [[0.2, 0.3], [0.1, 0.2]]
    resp = client.post('/ndvi-mask', json={'mode': 'bands', 'nir': nir, 'red': red})
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.data))
    assert img.size == (2, 2)


def test_invalid_ndvi_data(client):
    resp = client.post('/ndvi-mask', json={'ndvi': 'invalid'})
    assert resp.status_code == 400