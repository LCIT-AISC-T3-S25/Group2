from transformers import pipeline
from .utils import PassageIndex

# Load retriever
index = PassageIndex("app/passages.parquet")

# Load generator
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_answer(query: str) -> str:
    contexts = index.search(query)
    context_str = " ".join(contexts)
    input_text = f"question: {query} context: {context_str}"
    response = generator(input_text, max_length=200)[0]['generated_text']
    return response