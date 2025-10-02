import chromadb
# from langchain_huggingface import HuggingFaceEmbeddings
from inserting_file import load_pdf_to_strings, load_txt_to_strings
from jac_functions import chunk_research_paper, search_research_db, insert_publications
# from langchain_community.vectorstores import Chroma

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)
# Set up our embedding model
# vectorstore = Chroma(
#     client=client,
#     collection_name="ml_publications",
#     embedding_function=embeddings
# )
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
#     )

publication = load_pdf_to_strings("data/400 Level/1st Semester")
db = insert_publications(collection, publication, title="400 level")