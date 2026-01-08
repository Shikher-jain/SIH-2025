
AgroAI_Project
=============
Auto-generated project skeleton for HectFarm-1 sample pipeline.

Folders:
 - HectFarm-1/           put your cell_x folders here (satellite/sensor/weather JSON)
 - modules/              python modules (data loader, indices, model, etc.)
 - outputs/              where generated images, dashboards, predictions are stored

How to use:
1. Put your real data in AgroAI_Project\HectFarm-1 as described in the project.
2. Create a python venv and install requirements:
   python -m venv venv
   venv\Scripts\activate    (Windows)
   source venv/bin/activate   (Linux/Mac)
   pip install -r requirements.txt
3. Run main:
   python main.py

Notes:
 - This skeleton implements a CNN embedding + RF classifier flow, JSON advisory reports and heatmap dashboards.
 - Edit paths in main.py if your data sits elsewhere.
