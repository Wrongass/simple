import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from source.prompts import system_prompt  # Импортируем наш системный промпт
from source.tools import get_all_course_names, get_most_similar_course, register_for_course
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Проверка существования файла
env_path = Path(__file__).parent.parent / 'config' / 'demo_env.env'
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

def load_project_and_analysis(project_path: str, analysis_file: str):
    # 1. Загружаем все .java файлы Maven-проекта в виде документов
    loader = DirectoryLoader(project_path, glob="**/*.java", loader_cls=TextLoader)
    docs = loader.load()
    # docs – список Document, каждый содержит текст файла и метаданные (например, путь)

    # 2. Загружаем файл с аналитикой
    analysis_doc = TextLoader(analysis_file).load()[0]  # load() возвращает список из одного элемента

    # 3. Опционально: проверяем размер документов и при необходимости разбиваем их
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = []
    for doc in docs:
        if len(doc.page_content) > 2500:  # пример порога длины текста для разбиения
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                # Создаем новый Document для каждого фрагмента, сохраняя название файла в метаданных
                meta = doc.metadata.copy()
                meta["chunk"] = i
                split_docs.append(type(doc)(page_content=chunk, metadata=meta))
        else:
            split_docs.append(doc)
    # Теперь split_docs содержит либо оригинальные документы, либо разбитые на части большие файлы
    return split_docs, analysis_doc.page_content

def build_prompt_and_query(split_docs, analysis_text):
    # 1. Составляем системное сообщение с инструкцией
    system_content = (
        "Ты — AI-помощник, специалист по Java и автотестированию. "
        "Тебе предоставлен код проекта автотестов и обзор с предложениями по улучшению. "
        "Твоя задача: проанализировать текущий набор автотестов и сгенерировать новые тесты на Java (фреймворк TestNG) для улучшения покрытия. "
        "Сначала представь краткий анализ существующих тестов и проблем, затем предложи новые тесты. "
        "Новые тесты предоставь в виде полного кода Java-классов с использованием TestNG аннотаций @Test. "
        "Убедись, что новые тесты учитывают замечания из аналитического отчета и не дублируют уже существующие проверки."
    )
    system_message = SystemMessage(content=system_content)

    # 2. Компонуем пользовательское сообщение с данными проекта и аналитики
    user_content = "### Проект: исходные автотесты\n"
    for doc in split_docs:
        file_name = doc.metadata.get("source", "unknown file")
        # добавляем метку части, если файл был разделен
        if "chunk" in doc.metadata:
            file_name += f" (часть {doc.metadata['chunk'] + 1})"
        # Ограничиваем длину вставляемого текста каждого фрагмента, если нужно
        code_snippet = doc.page_content
        if len(code_snippet) > 1000:
            code_snippet = code_snippet[:1000] + "... [контент усечен]\n"
        user_content += f"\n=== Файл: {file_name} ===\n{code_snippet}\n"
    # Добавляем аналитический отчет
    user_content += "\n### Аналитический отчёт и предложения\n"
    user_content += analysis_text[:3000]  # ограничимся первыми N символами, если файл очень большой

    user_message = HumanMessage(content=user_content)

    # 3. Настраиваем подключение к модели GigaChat
    # (предполагается, что credentials уже заданы в переменной окружения GIGACHAT_CREDENTIALS)
    chat = model

    # 4. Вызываем модель с подготовленными сообщениями
    response = chat.invoke([system_message, user_message])
    return response.content  # текст ответа от GigaChat

def save_generated_tests(response_text: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    # Регулярное выражение для поиска блоков кода Java внутри ответа (```java ... ```)
    code_blocks = re.findall(r"```(?:java)?\s*(.*?)```", response_text, flags=re.DOTALL)
    if code_blocks:
        for i, code in enumerate(code_blocks, start=1):
            # Попытаемся определить имя класса из содержимого (по объявлению class)
            match = re.search(r"class\s+(\w+)", code)
            if match:
                class_name = match.group(1)
                file_name = f"{class_name}.java"
            else:
                file_name = f"GeneratedTest{i}.java"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code.strip() + "\n")
            saved_files.append(file_path)
    else:
        # Если блоков кода не нашли, сохраняем весь ответ как единый файл (на случай иного формата вывода)
        file_path = os.path.join(output_dir, "GeneratedTestOutput.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        saved_files.append(file_path)
    return saved_files

def generate_tests_for_project(project_path: str, analysis_file: str, output_dir: str):
    # todo: Передавать несколько файлов в виде "Аналитики".
    # 1. Загрузка контекста проекта и аналитики
    docs, analysis_text = load_project_and_analysis(project_path, analysis_file)
    # 2. Формирование промпта и запрос к GigaChat
    response_text = build_prompt_and_query(docs, analysis_text)
    # 3. Сохранение сгенерированных тестов в файлы
    result_files = save_generated_tests(response_text, output_dir)
    print(f"Сгенерировано файлов: {len(result_files)}. Они сохранены в папке: {output_dir}")
    return result_files

if __name__ == "__main__":
    print("🔐 Токен начинается с:", os.getenv("GIGACHAT_API_CREDENTIALS")[:10], "...")
    generate_tests_for_project(
        project_path="/Users/m_dev/Desktop/sberProject/simple/external_projects/coffee-autotests",
        analysis_file="/Users/m_dev/Desktop/sberProject/simple/external_projects/coffee-autotests/analysis.txt",
        output_dir="/Users/m_dev/Desktop/sberProject/simple/external_projects/coffee-autotests/generated_tests"
    )
    # chat('SberAX_consultant')