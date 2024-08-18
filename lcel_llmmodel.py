from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "与えられた言葉を{language}に変換してください"),
    HumanMessagePromptTemplate.from_template("hi there!")
])

chain = prompt | model | parser
result = chain.invoke({
    "language": "日本語"})

print(result)
# => こんにちは！
