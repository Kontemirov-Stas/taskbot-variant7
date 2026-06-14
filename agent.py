"""
agent.py — LangGraph-агент для TaskBot (Варіант 7)
Адаптовано з попередньої практичної роботи (Контеміров Станіслав, Варіант 7)
Модель: gpt-4o-mini через OpenAI API
"""

import re
import uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────
#  Схема стану графа
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # історія діалогу
    tasks: list[dict]                         # [{id, title, status, priority}]
    next_task_id: int                         # лічильник id


# ─────────────────────────────────────────────
#  Інструменти агента
# ─────────────────────────────────────────────
@tool
def create_task(title: str, tasks: list[dict], next_task_id: int) -> str:
    """Створює нову задачу зі статусом 'open'.
    Викликай коли юзер каже 'додай задачу', 'створи to-do', 'запиши що треба зробити'.
    """
    if not title.strip():
        return "Назва задачі не може бути порожньою."
    tasks.append({
        "id": next_task_id,
        "title": title,
        "status": "open",
        "priority": "🟡 Середній"
    })
    next_task_id += 1
    return f"✅ Створено задачу: '{title}' (id={next_task_id - 1})"


@tool
def set_task_done(task_id: int, tasks: list[dict]) -> str:
    """Позначає задачу як виконану за її id.
    Викликай для фраз типу 'познач задачу №2 як виконану', 'завершив задачу 3'.
    """
    for t in tasks:
        if t.get("id") == task_id:
            if t.get("status") == "done":
                return f"Задача {task_id} вже позначена як виконана."
            t["status"] = "done"
            return f"✅ Позначив задачу {task_id} ('{t['title']}') як виконану."
    return f"❌ Задачу з id={task_id} не знайдено."


@tool
def set_task_in_progress(task_id: int, tasks: list[dict]) -> str:
    """Позначає задачу як 'в процесі' за її id.
    Викликай коли юзер каже 'почав задачу X', 'беруся за задачу X', 'переведи в прогрес'.
    """
    for t in tasks:
        if t.get("id") == task_id:
            t["status"] = "in_progress"
            return f"🔄 Задача {task_id} ('{t['title']}') переведена в статус 'в процесі'."
    return f"❌ Задачу з id={task_id} не знайдено."


@tool
def list_open_tasks(tasks: list[dict]) -> str:
    """Повертає список невиконаних задач.
    Викликай коли юзер питає 'що залишилось', 'які задачі відкриті', 'покажи список'.
    """
    open_tasks = [t for t in tasks if t.get("status") != "done"]
    if not open_tasks:
        return "🎉 Всі задачі виконано!"
    lines = [
        f"{t['id']}. [{t.get('status','open')}] {t['title']} — {t.get('priority','🟡 Середній')}"
        for t in open_tasks
    ]
    return "📋 Відкриті задачі:\n" + "\n".join(lines)


@tool
def daily_summary(tasks: list[dict]) -> str:
    """Повертає щоденний підсумок: скільки виконано і скільки залишилось.
    Загальний огляд — зведення по всіх задачах.
    """
    if not tasks:
        return "📭 Задач поки немає."
    total = len(tasks)
    done = sum(1 for t in tasks if t.get("status") == "done")
    in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
    open_count = total - done - in_progress
    return (
        f"📊 Підсумок задач:\n"
        f"- 📋 Всього: {total}\n"
        f"- ✅ Виконано: {done}\n"
        f"- 🔄 В процесі: {in_progress}\n"
        f"- 📝 Відкритих: {open_count}"
    )


tools_list = [create_task, set_task_done, set_task_in_progress, list_open_tasks, daily_summary]


# ─────────────────────────────────────────────
#  Вузли графа
# ─────────────────────────────────────────────
def build_agent_node(llm_with_tools):
    """Фабрика вузла агента (closure щоб llm не бути глобальним)."""
    def agent_node(state: AgentState):
        tasks_info = state.get("tasks", [])
        next_id = state.get("next_task_id", 1)
        system_msg = (
            "Ти — розумний помічник для управління задачами (TaskBot). "
            "Відповідай ВИКЛЮЧНО українською мовою. "
            "Будь лаконічним та ввічливим. "
            f"Поточний список задач: {tasks_info}. "
            f"Наступний вільний id для нової задачі: {next_id}. "
            "Статуси задач: 'open' (нова), 'in_progress' (виконується), 'done' (готово). "
            "Використовуй відповідні інструменти для роботи з задачами. "
            "Якщо питання не стосується задач — відповідай як помічник."
        )
        msgs = [SystemMessage(content=system_msg)] + state["messages"]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}
    return agent_node


def build_tools_node_with_state():
    """Вузол інструментів, що синхронізує tasks між графом і session_state."""
    tool_node = ToolNode(tools_list)

    def tools_node(state: AgentState):
        result = tool_node.invoke(state)

        tasks = [dict(t) for t in state.get("tasks", [])]
        next_id = state.get("next_task_id", 1)

        for msg in result.get("messages", []):
            content = getattr(msg, "content", "")

            # Парсинг create_task
            if "Створено задачу" in content:
                m = re.search(r"'(.+)'.*id=(\d+)", content)
                if m:
                    title = m.group(1)
                    tid = int(m.group(2))
                    if not any(t["id"] == tid for t in tasks):
                        tasks.append({
                            "id": tid,
                            "title": title,
                            "status": "open",
                            "priority": "🟡 Середній"
                        })
                    next_id = max(next_id, tid + 1)

            # Парсинг set_task_done
            elif "як виконану" in content:
                m = re.search(r"задачу (\d+)", content)
                if m:
                    tid = int(m.group(1))
                    for t in tasks:
                        if t["id"] == tid:
                            t["status"] = "done"

            # Парсинг set_task_in_progress
            elif "в процесі" in content:
                m = re.search(r"Задача (\d+)", content)
                if m:
                    tid = int(m.group(1))
                    for t in tasks:
                        if t["id"] == tid:
                            t["status"] = "in_progress"

        result["tasks"] = tasks
        result["next_task_id"] = next_id
        return result

    return tools_node


# ─────────────────────────────────────────────
#  Ініціалізація графа
# ─────────────────────────────────────────────
def create_graph(openai_api_key: str):
    """Створює та компілює LangGraph граф.

    Args:
        openai_api_key: ключ OpenAI API

    Returns:
        Скомпільований граф із MemorySaver checkpointer
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )
    llm_with_tools = llm.bind_tools(tools_list)

    builder = StateGraph(AgentState)
    builder.add_node("agent", build_agent_node(llm_with_tools))
    builder.add_node("tools", build_tools_node_with_state())

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────
#  Функція виклику агента
# ─────────────────────────────────────────────
def invoke_agent(graph, user_message: str, thread_id: str) -> tuple[str, list[dict], int]:
    """Викликає агента і повертає (відповідь, tasks, next_task_id).

    Args:
        graph: скомпільований граф
        user_message: повідомлення від користувача
        thread_id: ідентифікатор сесії

    Returns:
        Кортеж (text_response, updated_tasks, next_task_id)
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )
    answer = result["messages"][-1].content
    tasks = result.get("tasks", [])
    next_id = result.get("next_task_id", 1)
    return answer, tasks, next_id


def get_state_tasks(graph, thread_id: str) -> tuple[list[dict], int]:
    """Отримує поточний стан задач з графа для заданого thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        tasks = state.values.get("tasks", [])
        next_id = state.values.get("next_task_id", 1)
        return tasks, next_id
    except Exception:
        return [], 1
