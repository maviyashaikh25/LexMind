import os
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_pinecone import PineconeVectorStore as PineconeVS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


# Load embeddings on CPU to avoid CUDA errors and device busy conflicts.
try:
    print("Initializing HuggingFaceEmbeddings on CPU...")
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT", model_kwargs={"device": "cpu"})
    print("HuggingFaceEmbeddings loaded successfully.")
except Exception as e:
    print(f"Error loading HuggingFaceEmbeddings: {e}. Trying default configuration.")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    except Exception as e2:
        print(f"Failed to load embeddings: {e2}")
        embeddings = None

vectorstore = None
if embeddings is not None:
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_ENV", "lexmind-cases")
        if pinecone_key and pinecone_key != "your_pinecone_key_here":
            print(f"Connecting to Pinecone index '{pinecone_index}'...")
            vectorstore = PineconeVS.from_existing_index(pinecone_index, embeddings)
            print("Pinecone connection established.")
        else:
            print("Pinecone API key is placeholder or missing. RAG vectorstore disabled.")
    except Exception as e:
        print(f"Failed to load Pinecone vectorstore: {e}. Falling back to mock RAG mode.")
        vectorstore = None
else:
    print("Embeddings could not be loaded. RAG vectorstore disabled.")


def get_retriever(court_filter: str = None, year_from: int = 2000):
    if vectorstore is None:
        return None
    search_kwargs = {"k": 20}
    if court_filter:
        search_kwargs["filter"] = {"court": {"$eq": court_filter}}
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )


PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""You are an expert Indian legal research assistant.

Use ONLY the retrieved case excerpts below to answer the question.
For every claim, cite the case name and year in parentheses.
If you are unsure, say "Based on available precedents, this is unclear."
Be concise — 3 to 5 sentences maximum.

Retrieved cases:
{summaries}

Question: {question}

Answer (with citations):"""
)


llm = None
# Check for Gemini key first, then Anthropic key
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

if gemini_key and gemini_key != "your_gemini_key_here":
    try:
        print("Initializing ChatGoogleGenerativeAI (Gemini)...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key,
            max_tokens=1000
        )
        print("Gemini LLM initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        llm = None

if llm is None and anthropic_key and anthropic_key != "your_anthropic_key_here":
    try:
        print("Initializing ChatAnthropic (Claude)...")
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=anthropic_key,
            max_tokens=1000
        )
        print("Anthropic LLM initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ChatAnthropic: {e}")
        llm = None

if llm is None:
    print("Neither Gemini nor Anthropic API keys are configured. LLM chain disabled.")

def get_rag_chain(court_filter=None):
    if vectorstore is None or llm is None:
        return None
    retriever = get_retriever(court_filter)
    if retriever is None:
        return None
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def run_rag(query: str, court: str = None) -> dict:
    if vectorstore is None or llm is None:
        # Return elegant mock response
        return {
            "answer": "Based on available precedents, the petitioner's argument has merit. Under similar circumstances, courts have held that procedural compliance is mandatory (State v. Sharma, 2018). (Note: This is a simulated RAG response as API keys are not configured.)",
            "sources": [
                {
                    "citation": "State v. Sharma (2018)",
                    "court": court or "Supreme Court",
                    "year": "2018",
                    "preview": "The court ruled that procedural requirements under Section 42 are mandatory and cannot be waived. Non-compliance results in the vitiation of the entire proceedings, as liberty is a fundamental right under Article 21."
                },
                {
                    "citation": "A.K. Gopalan v. State of Madras (1950)",
                    "court": "Supreme Court",
                    "year": "1950",
                    "preview": "Early precedent establishing the boundaries of personal liberty under Article 21, later expanded by Maneka Gandhi."
                }
            ]
        }
    
    try:
        chain  = get_rag_chain(court)
        if chain is None:
            raise ValueError("RAG chain could not be constructed.")
        result = chain.invoke({"question": query})
        sources = [
            {
                "citation": doc.metadata.get("citation", "Unknown"),
                "court": doc.metadata.get("court", ""),
                "year": doc.metadata.get("year", ""),
                "preview": doc.page_content[:300]
            }
            for doc in result["source_documents"][:5]
        ]
        return {"answer": result["answer"], "sources": sources}
    except Exception as e:
        print(f"Error executing RAG chain: {e}. Falling back to mock RAG response.")
        return {
            "answer": f"An error occurred while executing RAG chain: {e}. Showing mock precedents.",
            "sources": [
                {
                    "citation": "Fallback Case (2020)",
                    "court": court or "Supreme Court",
                    "year": "2020",
                    "preview": "This is a fallback case because the active RAG query failed with an exception."
                }
            ]
        }