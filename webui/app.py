# -*- coding: utf-8 -*-
"""
WebResearcher Web UI - FastAPI Application Entry
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import api_router

# 目录配置
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

# 确保数据目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 创建 FastAPI 应用
app = FastAPI(
    title="WebResearcher Deep Research Service",
    version="0.2.0",
    description="Web interface for WebResearcher - an intelligent research agent"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router)

# 挂载静态文件
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    """首页"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "webresearcher-webui"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
