"""Главный модуль для взаимодействия с агентом GigaChat."""

# Импорты
from source.agent import agent


# Основной цикл общения с агентом
def chat(thread_id: str):
    """
    Основная функция для общения с агентом.
    """
    config = {"configurable": {"thread_id": thread_id}}
    print("Добро пожаловать в терминал общения с GigaChat!")
    print("Напишите Ваш запрос или введите 'exit' для выхода.")

    while True:
        try:
            user_input = input("\n>>: ")
            if user_input.lower() == "exit":
                print("До свидания!")
                break

            response = agent.invoke({"messages": [("user", user_input)]}, config=config)
            print("🤖 :", response["messages"][-1].content)

        except KeyboardInterrupt:
            print("\nВыход из программы. До свидания!")
            break
        except Exception as e:
            print("Произошла ошибка:", str(e))


if __name__ == "__main__":
    chat('SberAX_consultant')