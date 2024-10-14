import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Download required NLTK packages
nltk.download('punkt')

# Step 1: Text Chunking (split large text into small chunks)
def chunk_text(text, chunk_size=1000):
    text = re.sub(r'\s+', ' ', text)  # Clean unnecessary whitespaces
    sentences = nltk.sent_tokenize(text)  # Sentence tokenization
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "

    if chunk:
        chunks.append(chunk.strip())

    return chunks

# Step 2: Text Retrieval using TF-IDF
class TextRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit(self, corpus):
        self.corpus = corpus
        self.corpus_embeddings = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=3):
        query_embedding = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]  # Get top-k most similar chunks
        return [self.corpus[i] for i in top_indices], similarities[top_indices]

# Step 3: Text Generation using T5 (Hugging Face Transformers)
class TextGenerator:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text):
        input_ids = self.tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Step 4: Retrieval-Augmented Generation (RAG) Logic
def generate_summary(text, query=None):
    # Step 4.1: Chunk the text
    chunks = chunk_text(text)

    # Step 4.2: Initialize the retriever and fit on chunks
    retriever = TextRetriever()
    retriever.fit(chunks)

    # Step 4.3: Retrieve relevant chunks (based on query or if no query, just take first chunk)
    if query:
        relevant_chunks, scores = retriever.retrieve(query)
    else:
        relevant_chunks = [chunks[0]]  # Default to the first chunk if no query

    # Step 4.4: Concatenate relevant chunks
    retrieved_text = " ".join(relevant_chunks)

    # Step 4.5: Initialize the text generator and generate the summary
    generator = TextGenerator(model_name='t5-small')  # Using T5-small for faster generation
    summary = generator.summarize(retrieved_text)

    return summary
