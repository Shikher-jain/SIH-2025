from src import app as _app
app = _app.app

with app.test_client() as c:
    resp = c.post('/ndvi-mask', json={'ndvi': [[0.1,0.5],[0.7,0.2]]})
    print('STATUS', resp.status_code)
    print(resp.get_data()[:1000])
