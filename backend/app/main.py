from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .models import ScriptRunRequest, ScriptRunResponse
from .script_registry import get_script, list_scripts

app = FastAPI(
    title="Script Control Center",
    version="0.1.0",
    description="统一管理并触发本地脚本的API服务",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Configure upload settings
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/scripts")
def available_scripts():
    return list_scripts()


@app.get("/api/scripts/{script_id}")
def get_script_metadata(script_id: str):
    script = get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="脚本不存在")
    return script.metadata


@app.post("/api/scripts/{script_id}/run", response_model=ScriptRunResponse)
async def run_script(script_id: str, payload: ScriptRunRequest):
    script = get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="脚本不存在")

    # 添加调试日志
    print(f"执行脚本: {script_id}")
    print(f"传递的参数: {payload.params}")

    try:
        result = await script.run(payload.params)
        print(f"脚本执行成功: {result}")
    except Exception as exc:  # pragma: no cover - safety net for runtime errors
        import traceback
        print(f"脚本执行失败: {exc}")
        print(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/upload/single")
async def upload_single_file(file: UploadFile = File(...)):
    """上传单个文件"""
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制 ({MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
        )

    # Generate unique filename
    if file.filename:
        # 检查文件是否已存在，如果存在才添加时间戳
        original_path = UPLOAD_DIR / file.filename
        if original_path.exists():
            file_ext = Path(file.filename).suffix
            base_name = Path(file.filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{base_name}_{timestamp}{file_ext}"
        else:
            unique_filename = file.filename
    else:
        # 如果没有文件名，生成一个带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}.xlsx"

    file_path = UPLOAD_DIR / unique_filename

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": file_size,
            "message": "文件上传成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")


@app.post("/api/upload/batch")
async def upload_batch_files(files: List[UploadFile] = File(...)):
    """批量上传文件"""
    results = []

    for file in files:
        try:
            # Check file size
            content = await file.read()
            file_size = len(content)

            if file_size > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"文件大小超过限制 ({MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
                })
                continue

            # Generate unique filename
            if file.filename:
                # 检查文件是否已存在，如果存在才添加时间戳
                original_path = UPLOAD_DIR / file.filename
                if original_path.exists():
                    file_ext = Path(file.filename).suffix
                    base_name = Path(file.filename).stem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{base_name}_{timestamp}{file_ext}"
                else:
                    unique_filename = file.filename
            else:
                # 如果没有文件名，生成一个带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{timestamp}.xlsx"

            file_path = UPLOAD_DIR / unique_filename

            # Save file
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            results.append({
                "filename": file.filename,
                "path": str(file_path),
                "size": file_size,
                "success": True,
                "message": "文件上传成功"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"上传失败: {str(e)}"
            })

    successful_uploads = [r for r in results if r.get("success", False)]
    failed_uploads = [r for r in results if not r.get("success", False)]

    return {
        "total": len(files),
        "successful": len(successful_uploads),
        "failed": len(failed_uploads),
        "results": results,
        "message": f"批量上传完成: {len(successful_uploads)}/{len(files)} 成功"
    }


@app.delete("/api/upload/{filename}")
async def delete_uploaded_file(filename: str):
    """删除上传的文件"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        os.remove(file_path)
        return {"message": "文件删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@app.get("/api/upload/list")
async def list_uploaded_files():
    """列出所有上传的文件"""
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created_time"], reverse=True)

        return {"files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")


class BatchDeleteRequest(BaseModel):
    filenames: List[str] = Field(default_factory=list, description="需要删除的文件名列表")


@app.post("/api/upload/batch-delete")
async def batch_delete_files(payload: BatchDeleteRequest):
    """批量删除上传的文件"""
    if not payload.filenames:
        raise HTTPException(status_code=400, detail="请提供需要删除的文件名列表")

    deleted = []
    failed = []

    for raw_name in payload.filenames:
        if not raw_name:
            failed.append({"filename": raw_name, "error": "文件名为空"})
            continue

        safe_name = Path(raw_name).name
        if safe_name != raw_name:
            failed.append({"filename": raw_name, "error": "非法文件名"})
            continue

        file_path = UPLOAD_DIR / safe_name
        if not file_path.exists():
            failed.append({"filename": raw_name, "error": "文件不存在"})
            continue

        try:
            file_path.unlink()
            deleted.append(safe_name)
        except OSError as exc:
            failed.append({"filename": raw_name, "error": str(exc)})

    return {
        "requested": len(payload.filenames),
        "deleted": deleted,
        "failed": failed,
        "message": f"已删除 {len(deleted)}/{len(payload.filenames)} 个文件",
    }


@app.get("/")
def root():
    return {
        "message": "服务运行中。访问 /api/scripts 获取脚本列表，或打开前端界面进行操作。",
        "docs": "/api/docs",
    }
