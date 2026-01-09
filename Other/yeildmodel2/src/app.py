from flask import Flask, request, jsonify
import numpy as np
from processing import (
    compute_ndvi_from_bands,
    ndvi_array_to_png,
)

app = Flask(__name__)

@app.route('/ndvi-mask', methods=['POST'])
def ndvi_mask():
    """Create an RGBA PNG mask from NDVI array or from NIR+Red bands.

    JSON body options:
    - mode: 'ndvi_array' (default) or 'bands'
    - ndvi: 2D list (if mode=='ndvi_array')
    - nir, red: 2D lists (if mode=='bands')
    - thresholds: {"low": float, "high": float}
    - yield_value: optional float to tweak thresholds
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    mode = data.get('mode', 'ndvi_array')
    thresholds = data.get('thresholds', {})
    low = float(thresholds.get('low', 0.33))
    high = float(thresholds.get('high', 0.66))
    yield_value = data.get('yield_value', None)

    try:
        if mode == 'bands':
            if 'nir' not in data or 'red' not in data:
                return jsonify({'error': 'nir and red bands are required for mode "bands"'}), 400
            nir = np.array(data['nir'], dtype=np.float32)
            red = np.array(data['red'], dtype=np.float32)
            if nir.shape != red.shape:
                return jsonify({'error': 'nir and red must have same shape'}), 400
            ndvi = compute_ndvi_from_bands(nir, red)
            png = ndvi_array_to_png(ndvi, thresholds=(low, high), yield_value=yield_value)
            return Response(png, mimetype='image/png')

        else:
            if 'ndvi' not in data:
                return jsonify({'error': 'ndvi is required for mode "ndvi_array"'}), 400
            ndvi = np.array(data['ndvi'], dtype=np.float32)
            if ndvi.ndim != 2:
                return jsonify({'error': 'ndvi must be a 2D array'}), 400
            png = ndvi_array_to_png(ndvi, thresholds=(low, high), yield_value=yield_value)
            return Response(png, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Processing error: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True)