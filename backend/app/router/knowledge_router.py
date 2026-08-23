import os

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.routing import APIRouter

from app.core.rate_limit import rate_limit
from app.core.success_response import success_response
from app.router.knowledge_service import KnowledgeService, get_knowledge_service
from app.schemas.models import (
    DocumentChunksResponse,
    KnowledgeDocumentDetail,
    KnowledgeListResponse,
    MD5ListResponse,
    MD5Record,
)
from app.utils.auth_utils import get_current_user_id

# 图片相关工具：定位存储目录，构建文件路径
from app.utils.image_extractor import get_image_storage_dir

knowledge_router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# 上传知识库
@knowledge_router.post("/add/single")
async def add_vector_single(
        file: UploadFile = File(...),

        # 鉴权
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=5, window=60))
):
    """上传文件，将文件保存到向量数据库，仅支持TXT和PDF"""
    filename = await knowledge_service.handle_add_vector_single(file, user_id)
    return success_response(message=f"文件 {filename} 已成功上传并存储到向量数据库")

# 上传单个文件，把文件解析、切片、向量化后写入 ChromaDB
@knowledge_router.post("/add/multiple")
async def add_vector_multiple(
        files: list[UploadFile] = File(..., description="要上传的文件列表，仅支持PDF和TXT格式"),
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=3, window=60))
):
    """上传多个文件，将文件保存到向量数据库，仅支持TXT和PDF"""
    filenames = await knowledge_service.handle_add_vector_multiple(files, user_id)
    return success_response(message=f"文件 {filenames} 已成功上传并存储到向量数据库")

# 批量上传多个文件，普通 JSON 返回。
"""
多文件时使用线程池，参数策略为文件数为切片任务数与 CPU 核心数两者的较小值，且强制至少为 1
切片的数进入队列由主线程完成入库
切片的业务属性为use_id,file_name,md5
"""
@knowledge_router.post("/add/multiple/stream")
async def add_vector_multiple_stream(
        files: list[UploadFile] = File(..., description="要上传的文件列表，仅支持PDF、TXT、MD、PPTX、DOCX格式"),
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=3, window=60))
):
    """上传多个文件，流式返回处理进度，仅支持TXT、PDF、MD、PPTX、DOCX"""
    # 内部链路（SSE 流式返回每一步进度）：
    #   1. 验证：_validate_and_read_files 校验 MIME/扩展名与总大小
    #   2. 切片：_start_slicing 用 ThreadPoolExecutor 并行切片
    #      └─ 每个工作线程执行 _sync_slice_file：
    #         a) get_file_document_sync 按扩展名分发到不同 loader 读取
    #         b) split_documents_sync  统一走 AsyncTextSplitter 切割
    #   3. 写入：_process_slice_results 串行消费切片队列 → ChromaDB.add_documents → 记录 MD5
    #
    # 不同文件类型读取差异（loader 层），切分阶段统一：
    #   .txt   → TextLoader                     整篇 → 1 个 Document
    #   .md    → UnstructuredMarkdownLoader      整篇 → 1 个 Document
    #   .pptx  → UnstructuredPowerPointLoader    整篇 → 1 个 Document
    #   .docx  → TextLoader（当前有缺陷，会读到二进制乱码）
    #   .pdf   → pdf_multimodal_loader（自研）   按页 → N 个 Document（含视觉描述）
    #   .pdf   → 回退 UnstructuredPDFLoader/PyPDFLoader  按页 → N 个 Document
    #
    # 所有类型读取后统一进入 AsyncTextSplitter：
    #   chunk_size=1000, chunk_overlap=50
    #   分隔符优先级: \n\n → \n → 。 → ！？!? → 空格 → 空字符串
    return StreamingResponse(
        knowledge_service.handle_add_vector_multiple_stream(files, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

# 2. 清空/删除知识库数据
@knowledge_router.delete("/clean")
async def clean_user_vectors(user_id: str = Depends(get_current_user_id), knowledge_service: KnowledgeService = Depends(get_knowledge_service)):
    """删除用户上传的所有向量"""
    await knowledge_service.clean_user_upload(user_id)
    return success_response(message="已成功删除用户上传的所有向量")

# 根据 MD5 删除某一个文档记录和对应向量
@knowledge_router.delete("/md5/clear")
async def clear_user_md5(
        delete_documents: bool = True,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    清空用户的MD5记录
    :param delete_documents: 是否同时删除知识库文档（默认True）
    """
    await knowledge_service.handle_clear_user_md5(user_id, delete_documents)
    if delete_documents:
        return success_response(message="已成功清空用户的MD5记录和知识库文档")
    else:
        return success_response(message="已成功清空用户的MD5记录（保留知识库文档）")

# 清空当前用户的 MD5 记录，可选择是否同时删除知识库文档。
@knowledge_router.delete("/md5/delete/{md5_value}")
async def delete_single_md5(
        md5_value: str,
        delete_documents: bool = True,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    删除单个MD5记录及其对应的知识库内容
    :param md5_value: 要删除的MD5值
    :param delete_documents: 是否同时删除知识库文档（默认True）
    """
    success = await knowledge_service.handle_delete_single_md5(user_id, md5_value, delete_documents)
    if success:
        if delete_documents:
            return success_response(message=f"已成功删除MD5记录 {md5_value} 及其对应的知识库文档")
        else:
            return success_response(message=f"已成功删除MD5记录 {md5_value}（保留知识库文档）")
    else:
        raise HTTPException(status_code=404, detail=f"MD5记录 {md5_value} 不存在")


@knowledge_router.delete("/delete/filename")
async def delete_by_filename(
        filename: str,
        delete_documents: bool = True,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    通过文件名删除MD5记录及其对应的知识库文档
    :param filename: 要删除的文件名
    :param delete_documents: 是否同时删除知识库文档（默认True）
    """
    success = await knowledge_service.handle_delete_by_filename(user_id, filename, delete_documents)
    if success:
        if delete_documents:
            return success_response(message=f"已成功删除文件 {filename} 的MD5记录及其对应的知识库文档")
        else:
            return success_response(message=f"已成功删除文件 {filename} 的MD5记录（保留知识库文档）")
    else:
        raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在")


@knowledge_router.get("/md5/list", response_model=MD5ListResponse)
async def get_all_md5_records(
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """获取用户的所有MD5记录"""
    records = await knowledge_service.handle_get_all_md5_records(user_id)
    return success_response(data=MD5ListResponse(
        records=records,
        total_count=len(records)
    ))


@knowledge_router.get("/md5/{md5_value}", response_model=MD5Record)
async def get_md5_info(
        md5_value: str,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """
    获取MD5对应的文档信息
    :param md5_value: MD5值
    """
    md5_info = await knowledge_service.handle_get_md5_info(user_id, md5_value)
    if md5_info:
        return success_response(data=md5_info)
    else:
        raise HTTPException(status_code=404, detail=f"MD5记录 {md5_value} 不存在")


@knowledge_router.get("/list", response_model=KnowledgeListResponse)
async def get_user_knowledge_list(
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """获取用户的知识库文档列表"""
    documents = await knowledge_service.handle_get_user_knowledge(user_id)
    return success_response(data=KnowledgeListResponse(
        documents=documents,
        total_count=len(documents)
    ))

# 获取当前用户的知识库文档列表，比如文件名、切片数量、预览内容。
@knowledge_router.get("/detail", response_model=KnowledgeDocumentDetail)
async def get_document_detail(
        filename: str,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """获取文档详情内容"""
    document = await knowledge_service.handle_get_document_detail(user_id, filename)
    return success_response(data=document)

# 获取某个文档的完整详情，包括内容、切片、图片等。
@knowledge_router.get("/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
        filename: str,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """获取文档切片信息"""
    chunks = await knowledge_service.handle_get_document_chunks(user_id, filename)
    return success_response(data=chunks)


# 图片服务端点：提供 PDF 中提取的原始图片的访问入口。
# 图片本身存储在服务器文件系统中，不直接对外暴露路径，而是通过此 API 做鉴权后返回。
# 这对安全性很重要——用户必须持有有效 JWT token 才能访问自己的图片。
@knowledge_router.get("/image/{md5}/{filename}")
async def serve_knowledge_image(
        md5: str,
        filename: str,
        user_id: str = Depends(get_current_user_id),
):
    """
    返回PDF中提取的原始图片（需JWT鉴权）
    图片存储在 data/extracted_images/{user_id}/{md5}/{filename}
    """
    image_dir = get_image_storage_dir(user_id, md5)
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图片不存在")

    # 根据文件扩展名设置正确的 Content-Type，确保浏览器正确渲染图片
    ext = os.path.splitext(filename)[1].lower()
    media_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.bmp': 'image/bmp',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    media_type = media_type_map.get(ext, 'application/octet-stream')
    return FileResponse(image_path, media_type=media_type)


# 批量图片获取接口：一次性拿到某个文档的所有图片，前端缓存后按需展示。
# 使用 base64 编码嵌入 JSON 中，减少前端的 HTTP 请求次数（尤其适合移动端）。
@knowledge_router.get("/images/all/{md5}")
async def serve_batch_images(
        md5: str,
        user_id: str = Depends(get_current_user_id),
        knowledge_service: KnowledgeService = Depends(get_knowledge_service),
        _: None = Depends(rate_limit(limit=10, window=60))
):
    """返回指定PDF的所有图片（单次请求，JSON + base64）"""
    result = await knowledge_service.handle_get_batch_images(user_id, md5)
    return success_response(data=result)
