import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import pinecone
from langchain_together import ChatTogether,TogetherEmbeddings


# Ensure you have the necessary environment variables set
# PINECONE_API_KEY, PINECONE_ENVIRONMENT, TOGETHER_API_KEY  
import os
# Set Together API key as environment variable (replace with your actual key)
os.environ["TOGETHER_API_KEY"] = "tgp_v1_Gdl66OKThh1KsJjEym9JEgDMqFWqd6bXtlZhviYqf34"

# Step 1: Read file from filepath and extract text using pypdf
def read_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Step 2: Use recursive text splitter to split the pdf data into small chunks
def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Step 3: Use bge-large embedding model and convert the chunks into embeddings
# Embedding method using langchain_together (ChatTogether) and BAAI/bge-base-en-v1.5
def get_embeddings_with_together(chunks):
    embedder = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        api_key=os.getenv("TOGETHER_API_KEY")
    )
    embeddings = []
    for chunk in chunks:
        # The embedding API expects a string, so we pass each chunk
        emb = embedder.embed_documents(chunk)
        print(f"Processed chunk: {chunk[:50]}...")  # Print first 50 characters of the chunk
        print(f"Embedding: {emb[:10]}...")  # Print first 10 values of the embedding
        embeddings.append(emb)
    # embeddings = embedder.embed_documents(chunks)
    # for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    #     print(f"Processed chunk {i}: {chunk[:50]}...")
    #     print(f"Embedding: {emb[:10]}...")
    return embeddings



# Step 4: Store the embeddings into Pinecone DB
def store_embeddings_pinecone(embeddings, chunks, index_name="pdf-embeddings"):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeddings[0]))
    index = pinecone.Index(index_name)
    vectors = [
        (f"chunk-{i}", emb, {"text": chunk})
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
    ]
    index.upsert(vectors)

if __name__ == "__main__":
    filepath = r"D:\Python_01\Python_01\Langchain01\Introduction_to_Data_and_Data_Science.pdf"  # Replace with your PDF file path
    text = read_pdf(filepath)
    chunks = split_text(text)
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Chunks: {chunks}")  # Print  chunk
    embeddings = get_embeddings_with_together(chunks)
    # store_embeddings_pinecone(embeddings, chunks)