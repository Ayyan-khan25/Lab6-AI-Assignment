from guided_chatbot import build_bot, reply

engine = build_bot()
assert "it_help.txt" in reply(engine, "How do I reset my password?")
assert reply(engine, "") == "Please enter a question."
assert "Possible matches" in reply(engine, "password recovery")
assert "No matching FAQ" in reply(engine, "volcano geology")
print("Task 1 FAQ behaviour checks passed.")
