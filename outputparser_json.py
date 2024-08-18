from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from pprint import pprint

model = ChatOpenAI(model="gpt-4o")

class TranslateResult(BaseModel):
    japanese: str = Field(description="日本語の結果")
    spanish: str = Field(description="スペイン語の結果")

parser = JsonOutputParser(pydantic_object=TranslateResult)

prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("与えられた言語に翻訳してください\n{format_instructions}"),
        HumanMessagePromptTemplate.from_template("{query}")
        ],
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    )

prompt = prompt_template.invoke({"query": "hello"})
# print(prompt)
# => messages=[SystemMessage(content='与えられた言語に翻訳してください\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"japanese": {"title": "Japanese", "description": "\\u65e5\\u672c\\u8a9e\\u306e\\u7d50\\u679c", "type": "string"}, "spanish": {"title": "Spanish", "description": "\\u30b9\\u30da\\u30a4\\u30f3\\u8a9e\\u306e\\u7d50\\u679c", "type": "string"}}, "required": ["japanese", "spanish"]}\n```'), HumanMessage(content='hello')]

output = model.invoke(prompt)
print(output)
# => content='```json\n{\n  "japanese": "こんにちは",\n  "spanish": "hola"\n}\n```' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 252, 'total_tokens': 274}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None} id='run-e59c414d-04a2-412a-82f0-1fcfe4e0cc6f-0' usage_metadata={'input_tokens': 252, 'output_tokens': 22, 'total_tokens': 274}

result = parser.invoke(output)
print(result)
# => {'japanese': 'こんにちは', 'spanish': 'hola'}