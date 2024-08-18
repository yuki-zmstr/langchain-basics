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

# See colab: https://colab.research.google.com/drive/1NO4lKg1z6wpI5EkApTwmXPfMnWKTTsXT#scrollTo=3TzmhkYZLWA8
# TODO: check time for larger list of documents
result_documents = db.similarity_search_with_relevance_scores(
    query="今日の晩ごはん", k=1)
print(result_documents)
print(result_documents[0])
