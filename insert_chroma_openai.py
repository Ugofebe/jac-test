import chromadb
from langchain_openai import OpenAIEmbeddings
from inserting_file import load_pdf_to_strings, load_txt_to_strings
from jac_functions import chunk_research_paper, search_research_db, insert_publications
from langchain_community.vectorstores import Chroma


def main():
    """Main function to initialize the database and insert publications."""
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./research_db")
    collection = client.get_or_create_collection(
        name="ml_publications",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Set up OpenAI embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    
    # Load publications and insert into database
    publication = load_pdf_to_strings("data/400 Level/1st Semester")
    db = insert_publications(collection, publication, title="400 level")


if __name__ == "__main__":
    main()