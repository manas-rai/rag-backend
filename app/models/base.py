from pydantic import BaseModel
from typing import List, Optional

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    error: Optional[str] = None

class DocumentBase(BaseModel):
    """Base document model."""
    content: str
    metadata: Optional[dict] = None 