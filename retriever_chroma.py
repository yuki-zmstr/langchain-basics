from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

documents = [
    Document(
        page_content="今日の晩御飯は、カレー？",
        metadata={"speaker": "son"}
    ),
    Document(
        page_content="今日の晩御飯、とんかつだよ",
        metadata={"speaker": "mother"}
    ),
    Document(
        page_content="カレーがよかったな",
        metadata={"speaker": "son"}
    )
]

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model
)

# Able to search in VectorStore, as well as Web site
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

result_documents = retriever.invoke("今日の晩御飯")
print(result_documents)
# => [Document(metadata={'speaker': 'mother'}, page_content='今日の晩御飯、とんかつだよ'), Document(metadata={'speaker': 'son'}, page_content='今日の晩御飯は、カレー？')]
