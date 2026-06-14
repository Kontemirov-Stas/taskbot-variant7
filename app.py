"""
app.py — TaskBot: Менеджер задач на базі Streamlit + LangGraph
Практичне завдання № 3 (тема 12) — Варіант 7
Автор: Контеміров Станіслав
"""

import json
import uuid
import streamlit as st

# ─────────────────────────────────────────────
#  Конфігурація сторінки (перший виклик Streamlit)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TaskBot — Менеджер задач",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Кастомні CSS-стилі
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Загальний фон */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8edf8 100%);
}

/* Kanban картки */
.task-card {
    background: white;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #6c8ebf;
    transition: transform 0.15s;
}
.task-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
.task-card-inprogress {
    border-left-color: #f0a500;
}
.task-card-done {
    border-left-color: #34a853;
    opacity: 0.75;
}

/* Kanban заголовки колонок */
.kanban-header {
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 14px;
}
.kanban-todo   { background: #dce8ff; color: #1a3a6b; }
.kanban-prog   { background: #fff3d0; color: #7a4d00; }
.kanban-done   { background: #d4f5e2; color: #1a5c38; }

/* Sidebar підзаголовки */
.sidebar-section {
    background: rgba(255,255,255,0.6);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 8px 0;
}

/* Повідомлення чату */
.stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Ініціалізація session_state
# ─────────────────────────────────────────────
def init_session():
    """Ініціалізує всі змінні сесії."""
    defaults = {
        "messages": [],           # [{role, content}] — для чат-інтерфейсу
        "tasks": [],              # [{id, title, status, priority}]
        "next_task_id": 1,        # лічильник id
        "thread_id": str(uuid.uuid4()),  # ідентифікатор LangGraph-сесії
        "graph": None,            # скомпільований граф
        "api_key_set": False,     # чи введено API-ключ
        "mode": "agent",          # "agent" або "chat"
        "system_prompt": (
            "Ти — розумний помічник для управління задачами. "
            "Відповідай українською мовою."
        ),
        "filter_status": "Всі",
        "filter_priority": "Всі",
        "token_count": 0,         # приблизна кількість токенів
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ─────────────────────────────────────────────
#  Допоміжні функції
# ─────────────────────────────────────────────
def get_tasks_by_status(status: str) -> list[dict]:
    """Повертає задачі за статусом з урахуванням фільтра пріоритету."""
    tasks = [t for t in st.session_state.tasks if t.get("status") == status]
    if st.session_state.filter_priority != "Всі":
        tasks = [t for t in tasks if t.get("priority") == st.session_state.filter_priority]
    return tasks


def add_task_direct(title: str, priority: str):
    """Додає задачу безпосередньо (без агента) через форму у sidebar."""
    tid = st.session_state.next_task_id
    st.session_state.tasks.append({
        "id": tid,
        "title": title,
        "status": "open",
        "priority": priority
    })
    st.session_state.next_task_id += 1


def set_status(task_id: int, new_status: str):
    """Змінює статус задачі кнопками Kanban."""
    for t in st.session_state.tasks:
        if t["id"] == task_id:
            t["status"] = new_status
            break


def delete_task(task_id: int):
    """Видаляє задачу зі списку."""
    st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task_id]


def count_tokens_approx(text: str) -> int:
    """Приблизний підрахунок токенів (1 токен ≈ 4 символи)."""
    return len(text) // 4


def export_chat_json() -> str:
    """Серіалізує історію чату у JSON."""
    return json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)


def export_chat_text() -> str:
    """Серіалізує історію чату у текст."""
    lines = []
    for msg in st.session_state.messages:
        role = "Ви" if msg["role"] == "user" else "TaskBot"
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def export_tasks_json() -> str:
    """Серіалізує задачі у JSON."""
    return json.dumps(st.session_state.tasks, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
#  Бічна панель (Sidebar)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 TaskBot")
    st.caption("Менеджер задач на базі LangGraph + Streamlit")
    st.divider()

    # ── API ключ ──────────────────────────────
    st.subheader("🔑 Налаштування API")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Необхідний для роботи агента (gpt-4o-mini)"
    )

    if api_key_input and not st.session_state.api_key_set:
        try:
            from agent import create_graph
            with st.spinner("Ініціалізація агента..."):
                st.session_state.graph = create_graph(api_key_input)
                st.session_state.api_key_set = True
            st.success("✅ Агент готовий!")
        except Exception as e:
            st.error(f"❌ Помилка: {e}")

    if st.session_state.api_key_set:
        st.success("🤖 Агент підключено")

    st.divider()

    # ── Режим роботи ─────────────────────────
    st.subheader("⚙️ Режим")
    mode = st.radio(
        "Режим чату:",
        ["🤖 Агент з інструментами", "💬 Звичайний чат"],
        index=0 if st.session_state.mode == "agent" else 1,
        help="Агент — використовує LangGraph та інструменти задач\nЧат — просто відповіді без інструментів"
    )
    st.session_state.mode = "agent" if "Агент" in mode else "chat"

    # ── Параметри генерації ──────────────────
    st.subheader("🎛️ Параметри")
    temperature = st.slider(
        "Температура (креативність)",
        min_value=0.0, max_value=1.0,
        value=0.0, step=0.1,
        help="0 = точні відповіді, 1 = більш творчі"
    )
    max_tokens = st.slider(
        "Макс. токенів відповіді",
        min_value=100, max_value=2000,
        value=500, step=100
    )

    st.divider()

    # ── Системний промпт ─────────────────────
    with st.expander("📝 Системний промпт", expanded=False):
        new_prompt = st.text_area(
            "Редагуйте промпт:",
            value=st.session_state.system_prompt,
            height=120
        )
        if st.button("💾 Зберегти промпт", use_container_width=True):
            st.session_state.system_prompt = new_prompt
            st.success("Збережено!")

    st.divider()

    # ── Швидке додавання задачі ──────────────
    st.subheader("➕ Нова задача")
    with st.form("new_task_form", clear_on_submit=True):
        new_title = st.text_input("Назва задачі", placeholder="Що треба зробити?")
        new_priority = st.selectbox(
            "Пріоритет",
            ["🔴 Високий", "🟡 Середній", "🟢 Низький"]
        )
        if st.form_submit_button("➕ Додати", use_container_width=True):
            if new_title.strip():
                add_task_direct(new_title.strip(), new_priority)
                st.success(f"Додано: {new_title[:30]}")
                st.rerun()
            else:
                st.warning("Введіть назву задачі!")

    st.divider()

    # ── Фільтри ──────────────────────────────
    st.subheader("🔍 Фільтри Kanban")
    st.session_state.filter_status = st.selectbox(
        "Статус",
        ["Всі", "Тільки відкриті", "В процесі", "Виконані"]
    )
    st.session_state.filter_priority = st.selectbox(
        "Пріоритет",
        ["Всі", "🔴 Високий", "🟡 Середній", "🟢 Низький"]
    )

    st.divider()

    # ── Статистика ────────────────────────────
    st.subheader("📊 Статистика")
    tasks = st.session_state.tasks
    total = len(tasks)
    done_count = sum(1 for t in tasks if t.get("status") == "done")
    prog_count = sum(1 for t in tasks if t.get("status") == "in_progress")
    open_count = total - done_count - prog_count

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("📋 Всього", total)
    col_s2.metric("✅ Виконано", done_count)
    col_s3, col_s4 = st.columns(2)
    col_s3.metric("🔄 В процесі", prog_count)
    col_s4.metric("📝 Відкритих", open_count)

    if total > 0:
        st.progress(done_count / total, text=f"Прогрес: {done_count}/{total}")

    st.metric("💬 Повідомлень", len(st.session_state.messages))
    st.metric("🔤 ~Токенів", st.session_state.token_count)

    st.divider()

    # ── Кнопки керування ─────────────────────
    st.subheader("🛠️ Керування")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🗑️ Очистити чат", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
    with col_b2:
        if st.button("♻️ Скинути задачі", use_container_width=True):
            st.session_state.tasks = []
            st.session_state.next_task_id = 1
            st.rerun()

    # Експорт
    st.download_button(
        "📥 Чат (JSON)",
        data=export_chat_json(),
        file_name="taskbot_chat.json",
        mime="application/json",
        use_container_width=True
    )
    st.download_button(
        "📄 Чат (TXT)",
        data=export_chat_text(),
        file_name="taskbot_chat.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.download_button(
        "📦 Задачі (JSON)",
        data=export_tasks_json(),
        file_name="taskbot_tasks.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()
    st.caption("ℹ️ TaskBot v1.0 | Варіант 7")
    st.caption(f"🆔 Thread: `{st.session_state.thread_id[:12]}...`")


# ─────────────────────────────────────────────
#  Заголовок
# ─────────────────────────────────────────────
st.markdown("# 📋 TaskBot — Менеджер задач")
st.caption(
    "Управляй задачами через чат або Kanban-дошку. "
    "Агент розуміє природну мову та автоматично оновлює дошку."
)

# ─────────────────────────────────────────────
#  Вкладки
# ─────────────────────────────────────────────
tab_chat, tab_kanban, tab_stats, tab_all = st.tabs([
    "💬 Чат з агентом",
    "📊 Kanban-дошка",
    "📈 Статистика",
    "📋 Всі задачі"
])


# ══════════════════════════════════════════════
#  ВКЛ 1 — ЧАТ
# ══════════════════════════════════════════════
with tab_chat:
    # Привітальне повідомлення
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Вітаю!** Я TaskBot — ваш помічник із задачами.\n\n"
                "Ось що я вмію:\n"
                "- ➕ **Додати задачу** — «Додай задачу: написати звіт»\n"
                "- ✅ **Позначити виконаною** — «Познач задачу 1 як виконану»\n"
                "- 🔄 **Взяти в роботу** — «Почав задачу 2»\n"
                "- 📋 **Переглянути список** — «Що в мене залишилось?»\n"
                "- 📊 **Підсумок** — «Підсумуй мої задачі»\n\n"
                "🔑 Введіть OpenAI API ключ у бічній панелі для роботи агента."
            )

    # Відображення історії чату
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Поле введення
    if prompt := st.chat_input("Напишіть команду або запитання..."):

        # Додаємо повідомлення користувача
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.token_count += count_tokens_approx(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Відповідь агента або звичайний чат
        with st.chat_message("assistant"):
            if st.session_state.mode == "agent" and st.session_state.api_key_set:
                # Режим агента — LangGraph
                with st.spinner("🤖 Агент думає..."):
                    try:
                        from agent import invoke_agent
                        answer, updated_tasks, next_id = invoke_agent(
                            st.session_state.graph,
                            prompt,
                            st.session_state.thread_id
                        )
                        # Синхронізуємо задачі зі стану графа
                        st.session_state.tasks = updated_tasks
                        st.session_state.next_task_id = next_id
                        st.session_state.token_count += count_tokens_approx(answer)
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                    except Exception as e:
                        err = f"❌ Помилка агента: {e}"
                        st.error(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err
                        })

            elif st.session_state.mode == "agent" and not st.session_state.api_key_set:
                msg_text = "⚠️ Введіть OpenAI API ключ у бічній панелі для активації агента."
                st.warning(msg_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg_text
                })

            else:
                # Звичайний чат без інструментів (заглушка)
                msg_text = (
                    "💬 Ви у режимі звичайного чату. "
                    "Переключіться на **«Агент з інструментами»** у бічній панелі, "
                    "щоб управляти задачами."
                )
                st.info(msg_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg_text
                })

        st.rerun()

    # Приклади діалогів (швидкий старт)
    st.markdown("---")
    st.markdown("**💡 Спробуй:**")
    example_cols = st.columns(4)
    examples = [
        "Додай задачу: написати звіт",
        "Познач задачу 1 як виконану",
        "Що в мене залишилось?",
        "Підсумуй мої задачі на сьогодні"
    ]
    for i, ex in enumerate(examples):
        with example_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                # Симулюємо введення
                st.session_state.messages.append({"role": "user", "content": ex})
                if st.session_state.api_key_set:
                    try:
                        from agent import invoke_agent
                        answer, updated_tasks, next_id = invoke_agent(
                            st.session_state.graph, ex, st.session_state.thread_id
                        )
                        st.session_state.tasks = updated_tasks
                        st.session_state.next_task_id = next_id
                        st.session_state.messages.append({
                            "role": "assistant", "content": answer
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ {e}"
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ Введіть API ключ для роботи агента."
                    })
                st.rerun()


# ══════════════════════════════════════════════
#  ВКЛ 2 — KANBAN
# ══════════════════════════════════════════════
with tab_kanban:
    st.subheader("📊 Kanban-дошка")

    # Визначаємо які задачі показувати
    all_tasks = st.session_state.tasks
    prio_filter = st.session_state.filter_priority

    def filtered(status_val):
        t = [x for x in all_tasks if x.get("status") == status_val]
        if prio_filter != "Всі":
            t = [x for x in t if x.get("priority") == prio_filter]
        return t

    todo_tasks = filtered("open")
    prog_tasks = filtered("in_progress")
    done_tasks = filtered("done")

    # Три колонки Kanban
    col_todo, col_prog, col_done = st.columns(3)

    # ── Колонка: До виконання ─────────────────
    with col_todo:
        st.markdown(
            f'<div class="kanban-header kanban-todo">📝 До виконання ({len(todo_tasks)})</div>',
            unsafe_allow_html=True
        )
        if not todo_tasks:
            st.info("Немає задач")
        for task in todo_tasks:
            with st.container(border=True):
                # Пріоритет-тег
                prio = task.get("priority", "🟡 Середній")
                st.markdown(
                    f"**{task['title']}**  \n"
                    f"<small>#{task['id']} · {prio}</small>",
                    unsafe_allow_html=True
                )
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("▶️ Взяти", key=f"prog_{task['id']}", use_container_width=True):
                        set_status(task["id"], "in_progress")
                        st.rerun()
                with btn2:
                    if st.button("✅ Готово", key=f"done_{task['id']}", use_container_width=True):
                        set_status(task["id"], "done")
                        st.rerun()
                # Вибір пріоритету
                new_prio = st.selectbox(
                    "Пріоритет",
                    ["🔴 Високий", "🟡 Середній", "🟢 Низький"],
                    index=["🔴 Високий", "🟡 Середній", "🟢 Низький"].index(
                        task.get("priority", "🟡 Середній")
                    ),
                    key=f"prio_{task['id']}",
                    label_visibility="collapsed"
                )
                if new_prio != task.get("priority"):
                    task["priority"] = new_prio
                    st.rerun()

                if st.button("🗑️", key=f"del_{task['id']}", use_container_width=True, help="Видалити"):
                    delete_task(task["id"])
                    st.rerun()

    # ── Колонка: В процесі ────────────────────
    with col_prog:
        st.markdown(
            f'<div class="kanban-header kanban-prog">🔄 В процесі ({len(prog_tasks)})</div>',
            unsafe_allow_html=True
        )
        if not prog_tasks:
            st.info("Немає задач")
        for task in prog_tasks:
            with st.container(border=True):
                prio = task.get("priority", "🟡 Середній")
                st.markdown(
                    f"**{task['title']}**  \n"
                    f"<small>#{task['id']} · {prio}</small>",
                    unsafe_allow_html=True
                )
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("◀️ Назад", key=f"back_{task['id']}", use_container_width=True):
                        set_status(task["id"], "open")
                        st.rerun()
                with btn2:
                    if st.button("✅ Готово", key=f"done2_{task['id']}", use_container_width=True):
                        set_status(task["id"], "done")
                        st.rerun()
                if st.button("🗑️", key=f"del2_{task['id']}", use_container_width=True, help="Видалити"):
                    delete_task(task["id"])
                    st.rerun()

    # ── Колонка: Виконано ─────────────────────
    with col_done:
        st.markdown(
            f'<div class="kanban-header kanban-done">✅ Виконано ({len(done_tasks)})</div>',
            unsafe_allow_html=True
        )
        if not done_tasks:
            st.info("Немає задач")
        for task in done_tasks:
            with st.container(border=True):
                prio = task.get("priority", "🟡 Середній")
                st.markdown(
                    f"~~{task['title']}~~  \n"
                    f"<small>#{task['id']} · {prio}</small>",
                    unsafe_allow_html=True
                )
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("↩️ Відкрити", key=f"reopen_{task['id']}", use_container_width=True):
                        set_status(task["id"], "open")
                        st.rerun()
                with btn2:
                    if st.button("🗑️ Видалити", key=f"del3_{task['id']}", use_container_width=True):
                        delete_task(task["id"])
                        st.rerun()


# ══════════════════════════════════════════════
#  ВКЛ 3 — СТАТИСТИКА
# ══════════════════════════════════════════════
with tab_stats:
    st.subheader("📈 Статистика задач")

    tasks = st.session_state.tasks
    if not tasks:
        st.info("📭 Задач ще немає. Додайте їх через чат або бічну панель.")
    else:
        total = len(tasks)
        done_n = sum(1 for t in tasks if t.get("status") == "done")
        prog_n = sum(1 for t in tasks if t.get("status") == "in_progress")
        open_n = total - done_n - prog_n

        # Метрики в рядок
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📋 Всього задач", total)
        m2.metric(
            "✅ Виконано",
            done_n,
            delta=f"{done_n/total*100:.0f}%" if total else None
        )
        m3.metric("🔄 В процесі", prog_n)
        m4.metric("📝 Відкритих", open_n)

        st.divider()

        # Прогрес-бар
        st.markdown("**Загальний прогрес виконання:**")
        st.progress(done_n / total if total else 0,
                    text=f"{done_n} з {total} виконано ({done_n/total*100:.0f}%)")

        st.divider()

        # Bar chart по статусах
        import pandas as pd

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("**Розподіл за статусами:**")
            status_data = {
                "Статус": ["📝 Відкриті", "🔄 В процесі", "✅ Виконані"],
                "Кількість": [open_n, prog_n, done_n]
            }
            df_status = pd.DataFrame(status_data).set_index("Статус")
            st.bar_chart(df_status)

        with col_chart2:
            st.markdown("**Розподіл за пріоритетом:**")
            high = sum(1 for t in tasks if t.get("priority") == "🔴 Високий")
            mid = sum(1 for t in tasks if t.get("priority") == "🟡 Середній")
            low = sum(1 for t in tasks if t.get("priority") == "🟢 Низький")
            prio_data = {
                "Пріоритет": ["🔴 Високий", "🟡 Середній", "🟢 Низький"],
                "Кількість": [high, mid, low]
            }
            df_prio = pd.DataFrame(prio_data).set_index("Пріоритет")
            st.bar_chart(df_prio)

        st.divider()

        # Деталі по пріоритетах
        with st.expander("🔴 Задачі з високим пріоритетом", expanded=True):
            high_tasks = [t for t in tasks if t.get("priority") == "🔴 Високий"]
            if high_tasks:
                for t in high_tasks:
                    status_icon = {"open": "📝", "in_progress": "🔄", "done": "✅"}.get(
                        t.get("status", "open"), "📝"
                    )
                    st.write(f"{status_icon} **#{t['id']}** — {t['title']}")
            else:
                st.write("Немає задач з високим пріоритетом")

        with st.expander("💬 Статистика чату"):
            msgs = st.session_state.messages
            user_msgs = [m for m in msgs if m["role"] == "user"]
            bot_msgs = [m for m in msgs if m["role"] == "assistant"]

            c1, c2, c3 = st.columns(3)
            c1.metric("👤 Повідомлень від вас", len(user_msgs))
            c2.metric("🤖 Відповідей агента", len(bot_msgs))
            c3.metric("🔤 Приблизно токенів", st.session_state.token_count)

            if bot_msgs:
                avg_len = sum(len(m["content"]) for m in bot_msgs) / len(bot_msgs)
                st.caption(f"Середня довжина відповіді агента: {avg_len:.0f} символів")


# ══════════════════════════════════════════════
#  ВКЛ 4 — ВСІ ЗАДАЧІ
# ══════════════════════════════════════════════
with tab_all:
    st.subheader("📋 Всі задачі")

    tasks = st.session_state.tasks

    if not tasks:
        st.info("📭 Задач немає. Скористайся чатом або формою в бічній панелі!")
    else:
        import pandas as pd

        # Фільтрація
        status_map = {
            "Всі": None,
            "Тільки відкриті": "open",
            "В процесі": "in_progress",
            "Виконані": "done"
        }
        filter_s = status_map.get(st.session_state.filter_status)

        display_tasks = tasks
        if filter_s:
            display_tasks = [t for t in tasks if t.get("status") == filter_s]
        if st.session_state.filter_priority != "Всі":
            display_tasks = [
                t for t in display_tasks
                if t.get("priority") == st.session_state.filter_priority
            ]

        st.caption(f"Показано: {len(display_tasks)} з {len(tasks)} задач")

        # DataFrame
        if display_tasks:
            status_labels = {
                "open": "📝 Відкрита",
                "in_progress": "🔄 В процесі",
                "done": "✅ Виконана"
            }
            df = pd.DataFrame([
                {
                    "ID": t["id"],
                    "Назва": t["title"],
                    "Статус": status_labels.get(t.get("status", "open"), "📝 Відкрита"),
                    "Пріоритет": t.get("priority", "🟡 Середній")
                }
                for t in display_tasks
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Завантаження таблиці
            st.download_button(
                "📥 Завантажити таблицю (CSV)",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="taskbot_tasks_table.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Немає задач за обраними фільтрами.")
