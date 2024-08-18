from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "与えた単語を{language}に変換して下さい"),
#     ("user", "Hello")
# ])

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("与えた単語を{language}に変換して下さい"),
    HumanMessage(content="Hello")
])

result = prompt_template.invoke({"language": "日本語"})
print(result)