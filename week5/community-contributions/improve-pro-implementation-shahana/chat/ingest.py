from pathlib import Path
# from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import re  # add this at the top of the file
from transformers import pipeline
# ADD these lines at the top, with your other imports/config
from sentence_transformers import SentenceTransformer
# Load once globally, not inside the function
# local_llm = pipeline(
#     "text-generation",
#     model="microsoft/Phi-3-mini-4k-instruct",  # small & capable
#     max_new_tokens=2000,
#     device_map="auto",  # uses GPU if available, else CPU
# )
import logging
import os

# At the top, after imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
load_dotenv(override=True)

# MODEL = "openai/gpt-4.1-nano"
# MODEL = "claude-haiku-3-5-20241022"

anthropic = Anthropic()
MODEL = "claude-haiku-4-5-20251001"

# After
# MODEL = "huggingface/mistralai/Mistral-7B-Instruct-v0.3"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
# embedding_model = "text-embedding-3-large"



EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent.parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100
wait = wait_exponential(multiplier=1, min=10, max=240)


WORKERS = 3

# openai = OpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with ONLY a valid JSON object. No markdown, no headings, no code fences, no explanation.
The JSON must follow this exact structure:
{{
    "chunks": [
        {{
            "headline": "short title",
            "summary": "few sentences summary",
            "original_text": "exact original text"
        }}
    ]
}}

"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]

# After
# def make_messages(document):
#     return [
#         {"role": "user", "content": make_prompt(document) + """
        
# You MUST respond with ONLY a valid JSON object, no extra text, no markdown, no code fences.
# The JSON must follow this exact structure:
# {
#     "chunks": [
#         {
#             "headline": "short title here",
#             "summary": "few sentences summary here",
#             "original_text": "exact original text here"
#         }
#     ]
# }
# """},
#     ]

@retry(wait=wait)
# def process_document(document):
#     messages = make_messages(document)
#     response = anthropic.messages.create(
#     model=MODEL,
#     max_tokens=8096,
#     messages=messages,
#     )
#     reply = response.content[0].text
#     doc_as_chunks = Chunks.model_validate_json(reply).chunks
#     return [chunk.as_result(document) for chunk in doc_as_chunks]


def process_document(document):
    log.info(f"START processing: {document['source']}")
    try:
        client = Anthropic()
        messages = make_messages(document)
        log.info(f"Calling API for: {document['source']}")
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            messages=messages,
        )
        reply = response.content[0].text
        log.info(f"Got response ({len(reply)} chars) for: {document['source']}")

        # Strip markdown code fences if present
        reply = re.sub(r"^```(?:json)?\s*", "", reply.strip())
        reply = re.sub(r"\s*```$", "", reply.strip())

        doc_as_chunks = Chunks.model_validate_json(reply).chunks
        log.info(f"Parsed {len(doc_as_chunks)} chunks for: {document['source']}")
        return [chunk.as_result(document) for chunk in doc_as_chunks]
    except Exception as e:
        log.error(f"FAILED for {document['source']}: {type(e).__name__}: {e}")
        raise RuntimeError(f"{type(e).__name__}: {e}") from None
# @retry(wait=wait)
# def process_document(document):
#     messages = make_messages(document)
#     prompt = messages[0]["content"]
    
#     output = local_llm(prompt)[0]["generated_text"]
    
#     # Strip the prompt itself from the output (local models echo the input)
#     reply = output[len(prompt):].strip()
#     reply = re.sub(r"```json|```", "", reply).strip()
    
#     doc_as_chunks = Chunks.model_validate_json(reply).chunks
#     return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


# def create_embeddings(chunks):
#     chroma = PersistentClient(path=DB_NAME)
#     if collection_name in [c.name for c in chroma.list_collections()]:
#         chroma.delete_collection(collection_name)

#     texts = [chunk.page_content for chunk in chunks]
#     emb = openai.embeddings.create(model=embedding_model, input=texts).data
#     vectors = [e.embedding for e in emb]

#     collection = chroma.get_or_create_collection(collection_name)

#     ids = [str(i) for i in range(len(chunks))]
#     metas = [chunk.metadata for chunk in chunks]

#     collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
#     print(f"Vectorstore created with {collection.count()} documents")

# BEFORE
# def create_embeddings(chunks):
#     chroma = PersistentClient(path=DB_NAME)
#     if collection_name in [c.name for c in chroma.list_collections()]:
#         chroma.delete_collection(collection_name)

#     texts = [chunk.page_content for chunk in chunks]
#     emb = openai.embeddings.create(model=embedding_model, input=texts).data  # <- OpenAI call
#     vectors = [e.embedding for e in emb]

#     collection = chroma.get_or_create_collection(collection_name)
#     ids = [str(i) for i in range(len(chunks))]
#     metas = [chunk.metadata for chunk in chunks]

#     collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
#     print(f"Vectorstore created with {collection.count()} documents")


# AFTER
def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.encode(texts, show_progress_bar=True).tolist()  # <- HuggingFace call

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
