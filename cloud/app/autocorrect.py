from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from oscillink.preprocess.autocorrect import smart_correct

from .config import get_settings
from .runtime_config import get_rate_limit

router = APIRouter()

_API_VERSION = get_settings().api_version


class AutoCorrectRequest(BaseModel):
    text: str
    custom_preserve: Optional[List[str]] = None


class AutoCorrectResponse(BaseModel):
    corrected: str
    meta: dict


@router.post(f"/{_API_VERSION}/autocorrect", response_model=AutoCorrectResponse)
def autocorrect(payload: AutoCorrectRequest, _rl=Depends(get_rate_limit)):
    corrected = smart_correct(payload.text, payload.custom_preserve)
    return AutoCorrectResponse(
        corrected=corrected,
        meta={"preserve_count": len(payload.custom_preserve or []), "len": len(corrected)},
    )
