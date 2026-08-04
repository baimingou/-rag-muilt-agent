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

ThinkingCallback = Callable[[dict], Awaitable[None]]


class StudyWorkflowState(TypedDict, total=False):
    query: str
    user_id: str
    context: dict[str, Any]
    plan: dict[str, Any]
    review: dict[str, Any]
    reviewer_feedback: str
    revision_count: int


def _extract_json(text: str) -> dict[str, Any]:
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
    try:
        return _extract_json(text)
    except Exception as exc:
        logger.error(f"解析学习 Agent JSON 失败: {exc}; raw={text[:500] if text else ''}")
        return fallback


def _compact_text(text: str | None, limit: int = 500) -> str:
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
        if self.thinking_callback:
            await self.thinking_callback({
                "type": "thinking",
                "stage": stage,
                "content": content,
                "details": details or {},
            })

    def _build_graph(self):
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
        user_id = state["user_id"]
        query = state["query"]
        await self._emit("collect_context", "正在收集待复习笔记、笔记统计和相关知识库资料...")

        await init_manager.models_ready.wait()
        await init_manager.note_service_ready.wait()

        async with AsyncSessionLocal() as db:
            today_reviews = await review_service.get_today_reviews(db, user_id)
            note_stats = await init_manager.note_service.get_category_stats(db, user_id)
            related_notes = await init_manager.note_service.search_notes(db, user_id, query, top_k=5)

        rag_docs = []
        try:
            rag_service = RagService(user_id, thinking_callback=self.thinking_callback)
            docs = await rag_service.retrieve_document(query)
            for doc in docs[:5]:
                source = doc.metadata.get("original_filename") or doc.metadata.get("source") or doc.metadata.get("title") or "unknown"
                rag_docs.append({
                    "source": str(source),
                    "source_type": doc.metadata.get("source_type", "knowledge_base"),
                    "preview": _compact_text(doc.page_content, 500),
                })
        except Exception as exc:
            logger.error(f"学习规划 RAG 上下文收集失败: {exc}", exc_info=True)

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
            "related_notes": [
                {
                    "note_id": note.id,
                    "title": note.title,
                    "category": note.category,
                    "tags": note.tags,
                    "content_preview": _compact_text(note.content, 350),
                    "source_ref": f"note:{note.id}",
                }
                for note in related_notes[:5]
            ],
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
            f"上下文收集完成：{len(context['today_reviews'])} 条待复习、{len(context['related_notes'])} 篇相关笔记、{len(context['knowledge_docs'])} 段知识库资料。",
            {
                "today_reviews": context["today_reviews"],
                "note_stats": note_stats,
            },
        )
        return {**state, "context": context}

    async def planner(self, state: StudyWorkflowState) -> StudyWorkflowState:
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
        initial_state: StudyWorkflowState = {
            "query": query,
            "user_id": user_id,
            "reviewer_feedback": "",
            "revision_count": 0,
        }
        result = await self.graph.ainvoke(initial_state)
        return {
            "query": query,
            "context": result.get("context", {}),
            "plan": result.get("plan", {}),
            "review": result.get("review", {}),
            "revision_count": result.get("revision_count", 0),
        }


def format_study_plan_markdown(result: dict[str, Any]) -> str:
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


async def get_study_plan_stream_response(query: str, session_id: str, user_id: str) -> AsyncGenerator[str, None]:
    thinking_queue = asyncio.Queue()
    result_holder: dict[str, Any] = {"result": None, "error": None}
    done = asyncio.Event()

    async def thinking_callback(data: dict):
        await thinking_queue.put(data)

    async def run_workflow():
        try:
            workflow = StudyPlanWorkflow(thinking_callback=thinking_callback)
            result_holder["result"] = await workflow.run(query, user_id)
        except Exception as exc:
            logger.error(f"学习规划工作流执行失败: {exc}", exc_info=True)
            result_holder["error"] = str(exc)
        finally:
            done.set()

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
