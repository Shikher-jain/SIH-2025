import numpy as np
import plotly.graph_objects as go

# Example: create a fake NDVI grid
rows, cols = 50, 50
ndvi = np.random.rand(rows, cols)

 
# Generate lat/lon grids (replace with real coordinates)
lat_grid = np.linspace(78, 25, rows)[:, None] * np.ones((rows, cols))
lon_grid = np.linspace(7, 80, cols)[None, :] * np.ones((rows, cols))

fig = go.Figure(go.Densitymapbox(
    lat=lat_grid.flatten(),
    lon=lon_grid.flatten(),
    z=ndvi.flatten(),
    radius=10,
    colorscale='RdYlGn',
    colorbar=dict(title='NDVI')
))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": lat_grid.mean(), "lon": lon_grid.mean()},
    mapbox_zoom=5,
    height=600,
    title="NDVI Map"
)

fig.show(renderer="browser")