from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def weather(place: str) -> int:
    "与えられた都市の天気をお伝えします"
    return "晴れです"


model = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = model.bind_tools([weather])

result = llm_with_tools.invoke("東京の天気を教えて")
print(result)
print(result.tool_calls)
