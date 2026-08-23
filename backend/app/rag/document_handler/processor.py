import asyncio
import os
import tempfile

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.logger_handler import logger
from app.rag.text_spliter import AsyncTextSplitter
from app.utils.config import chroma_config
from app.utils.file_handler import (
    get_file_md5_hex,
    listdir_allowed_type,
    markdown_loader,
    markdown_loader_sync,
    pdf_loader,
    pdf_loader_sync,
    ppt_loader,
    ppt_loader_sync,
    txt_loader,
    txt_loader_sync,
    word_loader,
    word_loader_sync,
)
from app.utils.pdf_multimodal_loader import pdf_multimodal_loader, pdf_multimodal_loader_sync


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, vectors_store: Chroma, md5_store, embed_model=None):
        self.vectors_store = vectors_store
        self.md5_store = md5_store
        self.spliter = AsyncTextSplitter(
            chunk_size=chroma_config['chunk_size'],
            chunk_overlap=chroma_config['chunk_overlap'],
            separators=chroma_config['separators'],
            embedding_model=embed_model
        )

    async def get_file_document(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """异步加载文件"""
        if read_path.endswith('.txt'):
            return await txt_loader(read_path)
        elif read_path.endswith('.pdf'):
            # 优先使用多模态加载器（提取图片+视觉描述），仅当提供了md5和user_id时才启用；
            # 这两个参数用于定位图片的存储路径 data/extracted_images/{user_id}/{md5}/
            if md5 and user_id:
                return await pdf_multimodal_loader(read_path, md5, user_id)
            # 回退到纯文本加载器（仅提取文字，无图片）
            return await pdf_loader(read_path)
        elif read_path.endswith('.md'):
            return await markdown_loader(read_path)
        elif read_path.endswith('.pptx'):
            return await ppt_loader(read_path)
        elif read_path.endswith('.docx'):
            return await word_loader(read_path)
        else:
            return []

    def get_file_document_sync(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """同步加载文件（用于多线程场景），按扩展名分发到不同 loader"""
        if read_path.endswith('.txt'):                      # .txt → TextLoader，整篇 → 1 个 Document（多编码尝试）
            return txt_loader_sync(read_path)
        elif read_path.endswith('.pdf'):                   # .pdf → 多模态加载器，按页 → N 个 Document
            if md5 and user_id:                             # 多模态需要 md5 定位图片存储路径 data/extracted_images/{user_id}/{md5}/
                return pdf_multimodal_loader_sync(read_path, md5, user_id)  # 含视觉描述，按页拆分
            return pdf_loader_sync(read_path)               # 回退：纯文本按页加载（无图片/视觉描述）
        elif read_path.endswith('.md'):                     # .md → UnstructuredMarkdownLoader(mode=single)，整篇 → 1 个 Document
            return markdown_loader_sync(read_path)
        elif read_path.endswith('.pptx'):                   # .pptx → UnstructuredPowerPointLoader(mode=single)，整篇 → 1 个 Document
            return ppt_loader_sync(read_path)
        elif read_path.endswith('.docx'):                   # .docx → TextLoader（当前有缺陷，无法解析 Word 二进制，读到乱码）
            return word_loader_sync(read_path)
        else:
            return []

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """同步分割文档（用于多线程场景），统一切分入口"""
        # 无论上游是哪种 loader 产出的 Document，都走同一个 AsyncTextSplitter
        # 参数来自 chroma.yaml: chunk_size=1000, chunk_overlap=50
        # 分隔符优先级: \n\n → \n → 。 → ！？!? → 空格 → 空字符串
        # 对每个 Document 独立切分；PDF 因 loader 层已按页拆分，此处对"每页内容"单独切，不会跨页拼接
        return self.spliter.split_documents_sync(documents)

    async def get_document(self, files: list = None, user_id: str = None, progress_callback=None):
        """
        处理文档并将其转为向量存入向量数据库
        :param files: 上传的文件列表，如果为None则从数据文件夹读取
        :param user_id: 用户ID，用于标记文档的所有者
        :param progress_callback: 进度回调函数，用于实时返回处理进度
        """
        file_paths = []
        file_names = {}

        if files:
            for file in files:
                temp_file_path = await asyncio.to_thread(
                    tempfile.NamedTemporaryFile,
                    delete=False,
                    suffix=os.path.splitext(file.filename)[1]
                )
                content = await file.read()
                await asyncio.to_thread(temp_file_path.write, content)
                file_paths.append(temp_file_path.name)
                file_names[temp_file_path.name] = file.filename
        else:
            allowed_file_path: tuple[str] = await listdir_allowed_type(
                chroma_config['data_path'],
                tuple(chroma_config['allow_knowledge_file_types'])
            )
            file_paths = list(allowed_file_path)

        for idx, file_path in enumerate(file_paths):
            filename = file_names.get(file_path, os.path.basename(file_path))

            md5_hex = await get_file_md5_hex(file_path)
            if await self.md5_store.check_md5_hex(md5_hex, user_id):
                if progress_callback:
                    await progress_callback({
                        'step': 'skipping',
                        'filename': filename,
                        'message': f'文件 {filename} 已存在，跳过'
                    })
                logger.info(f"【向量数据库】文件 {file_path} 的md5值 {md5_hex} 已存在，跳过")
                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
                continue

            try:
                if progress_callback:
                    await progress_callback({
                        'step': 'loading',
                        'filename': filename,
                        'message': f'正在加载文档 {filename}...'
                    })
                logger.info(f"【向量数据库】开始加载文档: {filename}")

                # 传入 md5_hex 和 user_id 以支持多模态PDF加载（图片提取和存储路径定位）
                document: list[Document] = await self.get_file_document(file_path, md5_hex, user_id)
                if not document:
                    if progress_callback:
                        await progress_callback({
                            'step': 'error',
                            'filename': filename,
                            'message': f'文件 {filename} 加载内容为空，跳过',
                            'error_message': '文件内容为空'
                        })
                    logger.error(f"【向量数据库】文件 {file_path} 加载内容为空，跳过")
                    if files:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                    continue

                if progress_callback:
                    await progress_callback({
                        'step': 'splitting',
                        'filename': filename,
                        'message': f'正在切分文档 {filename}...'
                    })
                logger.info(f"【向量数据库】开始切分文档: {filename}")

                document: list[Document] = await self.spliter.split_documents(document)
                if not document:
                    if progress_callback:
                        await progress_callback({
                            'step': 'error',
                            'filename': filename,
                            'message': f'文件 {filename} 切分内容为空，跳过',
                            'error_message': '文档切分后为空'
                        })
                    logger.error(f"【向量数据库】文件 {file_path} 切分内容为空，跳过")
                    if files:
                        try:
                            os.unlink(file_path)
                        except OSError:
                            pass
                    continue

                if progress_callback:
                    await progress_callback({
                        'step': 'storing',
                        'filename': filename,
                        'message': f'正在存储向量 {filename}...'
                    })
                logger.info(f"【向量数据库】开始存储向量: {filename}，文档数量: {len(document)}")

                if user_id:
                    for doc in document:
                        doc.metadata['user_id'] = user_id

                for doc in document:
                    doc.metadata['original_filename'] = filename
                    doc.metadata['md5'] = md5_hex

                await asyncio.to_thread(self.vectors_store.add_documents, document)

                original_filename = file_names.get(file_path, filename) if files else filename
                await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

                if progress_callback:
                    await progress_callback({
                        'step': 'completed',
                        'filename': filename,
                        'message': f'文件 {filename} 处理完成'
                    })
                logger.info(f"【向量数据库】文件 {file_path} 的md5值 {md5_hex} 已保存")

                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass

            except Exception as e:
                if progress_callback:
                    await progress_callback({
                        'step': 'error',
                        'filename': filename,
                        'message': f'文件 {filename} 处理失败',
                        'error_message': str(e)
                    })
                logger.error(f"【向量数据库】文件 {file_path} 处理时出错: {e}")
                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
                continue
