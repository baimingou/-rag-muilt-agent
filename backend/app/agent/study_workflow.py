import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.core.background_init import init_manager
from app.core.logger_handler import logger
from app.db.db_config import AsyncSessionLocal
from app.rag.rag_service import RagService
from app.services import session_manager as sm
from app.services.review_service import review_service
from app.utils.prompt_loader import load_prompt

# 学习计划模块采用 Planner-Reviewer 双 Agent 工作流：
# 1. collect_context 先把用户的学习数据收集成结构化上下文。
# 2. planner 只负责根据上下文生成可执行计划。
# 3. reviewer 只负责做质量检查，避免计划遗漏复习内容、缺少引用或过于空泛。
# 4. revise 根据 reviewer 的问题把反馈交回 planner，最多修正一次，防止无限循环。
# 这里用 LangGraph 的原因是：整个流程有共享状态、条件分支和回路，不是一次 LLM 调用能表达清楚的。
ThinkingCallback = Callable[[dict], Awaitable[None]]


class StudyWorkflowState(TypedDict, total=False):
    # 用户原始请求，例如“帮我安排今天的学习计划”。
    query: str
    # 当前登录用户 ID，用于隔离 MySQL 数据和 ChromaDB 向量检索结果。
    user_id: str
    # collect_context 节点产出的学习上下文，包含待复习笔记、知识库片段和统计信息。
    context: dict[str, Any]
    # planner 节点产出的结构化学习计划。
    plan: dict[str, Any]
    # reviewer 节点产出的评审报告，包括分数、问题、覆盖情况和修改建议。
    review: dict[str, Any]
    # reviewer 反馈给 planner 的修正意见。首次生成时为空，修正轮次才会填入。
    reviewer_feedback: str
    # 已经修正的次数。当前策略最多修正 1 次，避免 Agent 循环消耗过多 token。
    revision_count: int


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON。

    大模型有时会返回 ```json 代码块，或者在 JSON 前后附加解释文字。
    Planner 和 Reviewer 都要求结构化输出，所以这里做一次容错提取。
    """
    raw = (text or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    return json.loads(raw)


def _safe_json_loads(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """安全解析 JSON，解析失败时返回兜底结构，保证工作流不中断。"""
    try:
        return _extract_json(text)
    except Exception as exc:
        logger.error(f"解析学习 Agent JSON 失败: {exc}; raw={text[:500] if text else ''}")
        return fallback


def _compact_text(text: str | None, limit: int = 500) -> str:
    """压缩长文本，避免把整篇笔记或文档全部塞进 prompt 导致 token 过大。"""
    value = (text or "").replace("\n", " ").strip()
    return value[:limit]


class StudyPlanWorkflow:
    """Planner-Reviewer 双 Agent 学习规划工作流。"""

    def __init__(self, thinking_callback: ThinkingCallback | None = None):
        self.thinking_callback = thinking_callback
        self.planner_prompt = load_prompt("study_planner_prompt")
        self.reviewer_prompt = load_prompt("study_reviewer_prompt")
        self.graph = self._build_graph()

    async def _emit(self, stage: str, content: str, details: dict[str, Any] | None = None):
        """向前端推送 Agent 执行过程。

        前端 AIChat 通过 SSE 接收 type=thinking 的事件，
        因此用户可以看到“收集上下文 -> 生成计划 -> 评审 -> 修正”的实时过程。
        """
        if self.thinking_callback:
            await self.thinking_callback({
                "type": "thinking",
                "stage": stage,
                "content": content,
                "details": details or {},
            })

    def _build_graph(self):
        """构建 LangGraph 状态图。

        图结构：
            collect_context -> planner -> reviewer
                                      |
                                      v
                                finish / revise
                                           |
                                           v
                                        planner

        reviewer 后面使用条件边：
        - 评审通过或分数足够高：结束。
        - 不通过且还没修正过：进入 revise，再回到 planner 重新生成。
        """
        graph = StateGraph(StudyWorkflowState)
        graph.add_node("collect_context", self.collect_context)
        graph.add_node("planner", self.planner)
        graph.add_node("reviewer", self.reviewer)
        graph.add_node("revise", self.revise)

        graph.set_entry_point("collect_context")
        graph.add_edge("collect_context", "planner")
        graph.add_edge("planner", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self.should_revise,
            {
                "revise": "revise",
                "finish": END,
            },
        )
        graph.add_edge("revise", "planner")
        return graph.compile()

    async def collect_context(self, state: StudyWorkflowState) -> StudyWorkflowState:
        """收集学习计划所需的上下文。

        这里不是直接让 LLM “凭感觉”安排计划，而是先查真实数据：
        1. 今日待复习：MySQL review_records join notes，找 next_review_at <= now 的记录。
        2. 分类统计：MySQL notes 按 category 聚合，给 planner 一个用户知识分布概览。
        3. 知识库文档：RagService 走 HyDE + 向量检索，补充用户知识库里的外部资料。

        输出 context 会被 planner 和 reviewer 同时使用：
        - planner 用它生成计划。
        - reviewer 用它检查计划是否真的覆盖了待复习内容和引用来源。
        """
        user_id = state["user_id"]
        query = state["query"]
        await self._emit("collect_context", "正在收集待复习笔记、笔记统计和知识库补充资料...")

        await init_manager.models_ready.wait()
        await init_manager.note_service_ready.wait()

        # MySQL 查询部分：复习记录和笔记统计属于结构化业务数据，用 SQLAlchemy 查询最合适。
        async with AsyncSessionLocal() as db:
            today_reviews = await review_service.get_today_reviews(db, user_id)
            note_stats = await init_manager.note_service.get_category_stats(db, user_id)

        def _build_retrieval_query(
            base_query: str,
            reviews: list[dict[str, Any]],
            stats: dict[str, Any],
        ) -> str:
            parts: list[str] = []
            for item in reviews[:5]:
                title = (item.get("title") or "").strip()
                preview = (item.get("content_preview") or "").strip()
                if title:
                    parts.append(title)
                if preview:
                    parts.append(preview[:120])

            for item in (stats.get("categories") or [])[:5]:
                category = (item.get("category") or "").strip()
                if category:
                    parts.append(category)

            if base_query and base_query.strip():
                parts.append(base_query.strip())

            deduped: list[str] = []
            seen: set[str] = set()
            for part in parts:
                normalized = part.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    deduped.append(normalized)
            return "；".join(deduped) if deduped else base_query

        retrieval_query = _build_retrieval_query(query, today_reviews, note_stats)

        # RAG 检索部分：知识库资料不在 MySQL 里按关键词查，而是从 ChromaDB 向量库里找语义相关片段。
        rag_docs = []
        try:
            if not today_reviews:
                docs = []
            else:
                rag_service = RagService(user_id, thinking_callback=self.thinking_callback)
                docs = await rag_service.retrieve_document(retrieval_query)

            knowledge_candidates: list[dict[str, Any]] = []
            for doc in docs:
                if doc.metadata.get("source_type") == "note":
                    continue
                source = doc.metadata.get("original_filename") or doc.metadata.get("source") or doc.metadata.get("title") or "unknown"
                knowledge_candidates.append({
                    "source": str(source),
                    "source_type": doc.metadata.get("source_type", "knowledge_base"),
                    "preview": _compact_text(doc.page_content, 500),
                    "text": f"[来源: 知识库·{source}]\n{doc.page_content}",
                })

            if knowledge_candidates:
                reordered_texts = await rag_service.reorder_documents(
                    retrieval_query,
                    [item["text"] for item in knowledge_candidates],
                )
                text_to_item: dict[str, dict[str, Any]] = {}
                for item in knowledge_candidates:
                    text_to_item.setdefault(item["text"], item)

                for text in reordered_texts[:5]:
                    item = text_to_item.get(text)
                    if item:
                        rag_docs.append({
                            "source": item["source"],
                            "source_type": item["source_type"],
                            "preview": item["preview"],
                        })
            else:
                for item in knowledge_candidates[:5]:
                    rag_docs.append({
                        "source": item["source"],
                        "source_type": item["source_type"],
                        "preview": item["preview"],
                    })
        except Exception as exc:
            logger.error(f"学习规划 RAG 上下文收集失败: {exc}", exc_info=True)
        # 统一给每类来源生成 source_ref，后续 reviewer 会检查任务是否引用了这些来源。
        # source_ref 的格式固定为 review:<note_id>、knowledge:<source>。
        context = {
            "today_reviews": [
                {
                    "note_id": item.get("note_id"),
                    "title": item.get("title"),
                    "content_preview": item.get("content_preview"),
                    "review_count": item.get("review_count"),
                    "last_reviewed_at": item.get("last_reviewed_at"),
                    "interval_days": item.get("interval_days"),
                    "source_ref": f"review:{item.get('note_id')}",
                }
                for item in today_reviews[:8]
            ],
            "related_notes": [],
            "knowledge_docs": [
                {
                    **doc,
                    "source_ref": f"knowledge:{doc['source']}",
                }
                for doc in rag_docs
            ],
            "note_stats": note_stats,
        }

        await self._emit(
            "collect_context",
            f"上下文收集完成：{len(context['today_reviews'])} 条待复习、{len(context['knowledge_docs'])} 段知识库补充资料。",
            {
                "today_reviews": context["today_reviews"],
                "note_stats": note_stats,
                "retrieval_query": retrieval_query,
            },
        )
        return {**state, "context": context}

    async def planner(self, state: StudyWorkflowState) -> StudyWorkflowState:
        """Study Planner Agent：负责生成计划，不负责质检。

        输入：
        - 用户请求 query
        - collect_context 产出的结构化上下文
        - reviewer_feedback，首次为空，修正时包含 reviewer 的问题和建议

        输出：
        - summary：计划概览
        - tasks：具体任务，必须带 action/reason/source_refs/estimated_minutes/priority
        - weak_points：薄弱点
        - suggested_new_notes：建议补充的新笔记
        """
        await self._emit("planner", "Study Planner Agent 正在生成学习计划...")
        prompt = self.planner_prompt.format(
            query=state["query"],
            context=json.dumps(state["context"], ensure_ascii=False, indent=2),
            reviewer_feedback=state.get("reviewer_feedback", "无"),
        )
        response = await init_manager.chat_model.ainvoke([HumanMessage(content=prompt)])
        plan = _safe_json_loads(
            response.content,
            {
                "summary": "学习计划生成失败，请稍后重试。",
                "tasks": [],
                "weak_points": [],
                "suggested_new_notes": [],
            },
        )
        await self._emit("planner", f"学习计划生成完成，共 {len(plan.get('tasks', []))} 个任务。", {"plan": plan})
        return {**state, "plan": plan}

    async def reviewer(self, state: StudyWorkflowState) -> StudyWorkflowState:
        """Reviewer Agent：负责检查 planner 结果。

        Reviewer 不重新写计划，只做质量门控，主要看三类问题：
        1. coverage：today_reviews 里的待复习笔记有没有被覆盖。
        2. citation：每个任务有没有 source_refs，且引用是否来自当前上下文。
        3. specificity：任务是否具体可执行，是否只是“复习知识点”这种空话。

        这样做的价值是把“生成”和“评审”分离，减少单个 prompt 自说自话的问题。
        """
        await self._emit("reviewer", "Reviewer Agent 正在检查覆盖率、引用来源和任务具体性...")
        prompt = self.reviewer_prompt.format(
            query=state["query"],
            context=json.dumps(state["context"], ensure_ascii=False, indent=2),
            plan=json.dumps(state["plan"], ensure_ascii=False, indent=2),
        )
        response = await init_manager.chat_model.ainvoke([HumanMessage(content=prompt)])
        review = _safe_json_loads(
            response.content,
            {
                "passed": False,
                "score": 0,
                "issues": [{"type": "other", "message": "Reviewer 输出解析失败", "target": "reviewer"}],
                "revision_suggestions": ["重新生成更具体且带来源引用的学习计划"],
                "coverage": {"covered_review_note_ids": [], "missing_review_note_ids": []},
            },
        )
        await self._emit(
            "reviewer",
            f"Reviewer 检查完成：得分 {review.get('score', 0)}，{'通过' if review.get('passed') else '需要修正'}。",
            {"review": review},
        )
        return {**state, "review": review}

    async def revise(self, state: StudyWorkflowState) -> StudyWorkflowState:
        """把 Reviewer 的问题整理成 planner 可读的修正反馈。

        revise 节点本身不调用 LLM，它只负责把 review 中的 score、issues、coverage、
        revision_suggestions 打包到 reviewer_feedback。下一轮 planner 会把这些反馈放进 prompt，
        从而生成更具体、更完整、更有来源引用的学习计划。
        """
        review = state.get("review", {})
        suggestions = review.get("revision_suggestions", [])
        feedback = json.dumps(
            {
                "score": review.get("score"),
                "issues": review.get("issues", []),
                "revision_suggestions": suggestions,
                "coverage": review.get("coverage", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        revision_count = state.get("revision_count", 0) + 1
        await self._emit("revise", "Reviewer 未通过，正在把反馈交给 Planner 进行一次自我修正...", {"feedback": feedback})
        return {**state, "reviewer_feedback": feedback, "revision_count": revision_count}

    def should_revise(self, state: StudyWorkflowState) -> str:
        """Reviewer 后的条件分支。

        分支策略：
        - passed=true 或 score>=85：认为计划可用，直接结束。
        - 不通过且 revision_count=0：进入 revise，再回 planner 修正一次。
        - 已经修正过一次：无论分数如何都结束，避免无限循环和过高成本。
        """
        review = state.get("review", {})
        score = int(review.get("score") or 0)
        passed = bool(review.get("passed"))
        revision_count = state.get("revision_count", 0)
        if revision_count >= 1:
            return "finish"
        if passed or score >= 85:
            return "finish"
        return "revise"

    async def run(self, query: str, user_id: str) -> dict[str, Any]:
        """执行完整学习规划工作流，并返回最终状态。"""
        # 构造初始状态，输入给 LangGraph
        initial_state: StudyWorkflowState = {
            "query": query,
            "user_id": user_id,
            "reviewer_feedback": "",
            "revision_count": 0,
        }
        # 异步执行LangGraph图，ainvoke = async invoke
        result = await self.graph.ainvoke(initial_state)
        # 从图输出的state中提取字段，组装对外返回的字典
        return {
            "query": query,
            "context": result.get("context", {}),
            "plan": result.get("plan", {}),
            "review": result.get("review", {}),
            "revision_count": result.get("revision_count", 0),
        }


def format_study_plan_markdown(result: dict[str, Any]) -> str:
    """把 planner/reviewer 的结构化 JSON 转成前端聊天页可直接渲染的 Markdown。"""
    plan = result.get("plan", {})
    review = result.get("review", {})
    context = result.get("context", {})

    lines = ["## 学习计划", ""]
    lines.append(plan.get("summary") or "已生成学习计划。")
    lines.append("")

    tasks = plan.get("tasks", [])
    if tasks:
        lines.append("### 今日任务")
        for index, task in enumerate(tasks, 1):
            refs = ", ".join(task.get("source_refs", [])) or "无"
            lines.append(f"{index}. **{task.get('title', '未命名任务')}**")
            lines.append(f"   - 动作：{task.get('action', '')}")
            lines.append(f"   - 原因：{task.get('reason', '')}")
            lines.append(f"   - 时间：{task.get('estimated_minutes', 0)} 分钟")
            lines.append(f"   - 优先级：{task.get('priority', 'medium')}")
            lines.append(f"   - 来源：{refs}")
    else:
        lines.append("暂无可执行任务。")

    weak_points = plan.get("weak_points", [])
    if weak_points:
        lines.extend(["", "### 薄弱点"])
        for item in weak_points:
            lines.append(f"- {item}")

    suggested = plan.get("suggested_new_notes", [])
    if suggested:
        lines.extend(["", "### 建议补充的笔记"])
        for item in suggested:
            lines.append(f"- **{item.get('title', '未命名笔记')}**：{item.get('reason', '')}")

    lines.extend(["", "### Reviewer 检查"])
    lines.append(f"- 得分：{review.get('score', 0)}")
    lines.append(f"- 状态：{'通过' if review.get('passed') else '已修正/仍需关注'}")
    if result.get("revision_count", 0):
        lines.append(f"- 自我修正：{result['revision_count']} 次")

    issues = review.get("issues", [])
    if issues:
        lines.append("- 发现的问题：")
        for issue in issues:
            lines.append(f"  - [{issue.get('type', 'other')}] {issue.get('message', '')}")

    today_count = len(context.get("today_reviews", []))
    lines.extend(["", f"> 本次计划参考了 {today_count} 条待复习记录。"])
    return "\n".join(lines)


async def get_study_plan_stream_response(
        query: str, session_id: str, user_id: str) -> AsyncGenerator[str, None]:
    """学习计划 SSE 接口的响应生成器。

    工作流在后台 task 中执行，thinking_queue 用来实时转发每个节点的进度。
    最终结果会：
    1. 格式化成 Markdown。
    2. 写入会话历史，保证刷新页面后仍能看到本次计划。
    3. 按 chunk 流式推给前端，和普通 AIChat 的体验保持一致。
    """
    # 创建一个异步队列，专门用来存放思考过程/中间输出
    thinking_queue = asyncio.Queue()

    # 结果容器字典，保存最终返回结果和异常信息
    result_holder: dict[str, Any] = {"result": None, "error": None}

    # 异步事件对象，用来做任务完成的信号通知
    done = asyncio.Event()

    async def thinking_callback(data: dict):
        await thinking_queue.put(data)

    async def run_workflow():
        try:
            # 实例化化工作流的类
            workflow = StudyPlanWorkflow(thinking_callback=thinking_callback)

            # 调用LangGraph执行学习规划工作流，再把图输出的内部状态整理成对外业务字典返回最终学习规划结果
            result_holder["result"] = await workflow.run(query, user_id)
        except Exception as exc:
            logger.error(f"学习规划工作流执行失败: {exc}", exc_info=True)
            result_holder["error"] = str(exc)
        finally:
            done.set()
                # 异步执行
    task = asyncio.create_task(run_workflow())

    try:
        yield f"data: {json.dumps({'type': 'response', 'content': '', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        while not done.is_set() or not thinking_queue.empty():
            try:
                event = await asyncio.wait_for(thinking_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                thinking_queue.task_done()
            except TimeoutError:
                continue

        await task

        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'content': result_holder['error'], 'session_id': session_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            return

# 格式化输出内容
        content = format_study_plan_markdown(result_holder["result"])


        await sm.session_manager.add_message(session_id, user_id, query, content)
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            yield f"data: {json.dumps({'type': 'response', 'content': content[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
    except Exception as exc:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc), 'session_id': session_id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
