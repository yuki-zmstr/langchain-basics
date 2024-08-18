from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

a = embeddings_model.embed_query("Hello")
b = embeddings_model.embed_query("こんにちは")
c = embeddings_model.embed_query("こんばんは。")
d = embeddings_model.embed_query("明日のご飯はカレーです！")

print("number of dimensions", len(a))
print("first three coordinates of a vector", a[:3])
print("a,b cosine similarity", cosine_similarity([a], [b]))
print("a,c cosine similarity", cosine_similarity([a], [c]))
print("a,d cosine similarity", cosine_similarity([a], [d]))
