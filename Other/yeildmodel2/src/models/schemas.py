from pydantic import BaseModel
from typing import List, Optional

class NDVIRequest(BaseModel):
    ndvi_values: List[float]
    threshold: Optional[float] = 0.3

class YieldRequest(BaseModel):
    yield_values: List[float]
    threshold: Optional[float] = 0.5

class MaskResponse(BaseModel):
    mask: List[int]  # 0 for transparent, 1 for red, 2 for yellow, 3 for green
    message: str