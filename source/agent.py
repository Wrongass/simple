import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from source.prompts import system_prompt  # Импортируем наш системный промпт
from source.tools import get_all_course_names, get_most_similar_course, register_for_course
# Проверка существования файла
env_path = Path('./config/demo_env.env')
if not env_path.exists():
    print(f"❌ Файл .env не найден по пути: {env_path.absolute()}")
else:
    print(f"✅ Файл .env найден: {env_path.absolute()}")
    # Загрузка переменных окружения
    load_dotenv(env_path)

# Инициализация модели GigaChat
model = GigaChat(
    credentials=os.getenv("GIGACHAT_API_CREDENTIALS"),
    scope=os.getenv("GIGACHAT_API_SCOPE"),
    model=os.getenv("GIGACHAT_MODEL_NAME"),
    verify_ssl_certs=False,
    profanity_check=False,
    timeout=600,
    top_p=0.3,
    temperature=0.1,
    max_tokens=6000
)

# response = model.invoke([HumanMessage(content="Привет, какой посоветуешь курс?")])
# console = Console()
# console.print(response)

tools = [get_all_course_names, get_most_similar_course, register_for_course]
agent = create_react_agent(
    model=model,
    tools=tools,
    state_modifier=system_prompt,  # Подключаем системный контекст
    checkpointer=MemorySaver()  # Добавляем объект из библиотеки LangGraph для сохранения памяти агента
)

def chat(thread_id: str):
    print("Тест работы агента с системным промптом.")

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": "Я ИТ-архитектор. Какие курсы мне подойдут?",}
        ]},
        config = {"configurable": {"thread_id": thread_id}}
    )
    print("Ответ консультанта:", response["messages"][-1].content)

if __name__ == "__main__":
    chat('SberAX_consultant')