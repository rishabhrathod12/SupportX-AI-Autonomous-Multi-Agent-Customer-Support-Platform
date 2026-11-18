"""A small local demo page (customer chat + staff approvals). Never mounted in production."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGE = Path(__file__).resolve().parents[1] / "static" / "demo.html"

router = APIRouter(include_in_schema=False)


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
