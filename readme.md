# GraphRAG — Graph-Enhanced Retrieval-Augmented Generation

An industry-standard RAG pipeline that combines **semantic chunking**, **cross-encoder reranking**, and **knowledge graph traversal** to answer questions from PDF documents. Built with Groq (Llama 3.3), FAISS, and NetworkX.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION (offline, runs once)            │
│                                                             │
│  PDF → Semantic Chunking → FAISS Vector Store → Persist     │
│                        └──→ Knowledge Graph   → Persist     │
│              (spaCy NER + LLM concept extraction)           │
└─────────────────────────────────────────────────────────────┘
                              │
                    indexes/faiss/  &  indexes/knowledge_graph.pkl
                              │
┌─────────────────────────────────────────────────────────────┐
│                    QUERY (online, per request)               │
│                                                             │
│  [1] Query Rewriting          (LLM reformulates query)      │
│  [2] Vector Retrieval         (FAISS top-k)                 │
│  [3] Cross-Encoder Reranking  (ms-marco-MiniLM reranker)    │
│  [4] Graph Expansion          (Dijkstra traversal)          │
│  [5] Answer Generation        (single LLM call)             │
└─────────────────────────────────────────────────────────────┘
```

## What Makes This Different from Basic RAG

| Feature | Basic RAG | This Project |
|---|---|---|
| Chunking | Fixed-size (RecursiveCharacterTextSplitter) | **Semantic chunking** (splits at meaning boundaries) |
| Retrieval | Vector similarity only | Vector + **knowledge graph traversal** |
| Reranking | None | **Cross-encoder reranking** (ms-marco-MiniLM) |
| Query | Raw user query | **LLM-rewritten query** before retrieval |
| Persistence | Rebuilt every run | **FAISS + graph persisted to disk** |
| Context Assembly | Flat list of top-k chunks | **Dijkstra traversal** discovers conceptually connected chunks |

## How Graph Expansion Works

After vector retrieval + reranking gives the top chunks, the system:

1. Maps those chunks to nodes in the pre-built knowledge graph
2. Runs a **Dijkstra-like traversal** following edges weighted by:
   - Cosine similarity between chunk embeddings (70%)
   - Shared concepts between chunks (30%)
3. Discovers **neighboring chunks** that are conceptually related but wouldn't rank high in vector search alone
4. Assembles the traversal-ordered context for the LLM

![Graph Traversal](graph_traversal.png)

## Setup

```bash
# Clone
git clone https://github.com/amruth6002/GraphRAG.git
cd GraphRAG

# Virtual environment
python -m venv newenv
source newenv/bin/activate

# Install dependencies
pip install langchain langchain-groq langchain-huggingface langchain-community langchain-experimental
pip install faiss-cpu sentence-transformers
pip install networkx scikit-learn spacy nltk matplotlib tqdm pydantic
pip install python-dotenv pypdf
python -m spacy download en_core_web_sm

# Environment variables
echo 'GROQ_API_KEY="your-groq-api-key"' > .env
```

## Usage

```bash
# First run: ingests PDF + builds indexes + answers query
python src/ingestion.py --path data/Understanding_Climate_Change.pdf \
    --query "what is the main cause of climate change?"

# Subsequent runs: loads from disk (fast startup) + answers query
python src/ingestion.py --query "what are the effects of deforestation?"

# Force rebuild indexes
python src/ingestion.py --path data/Understanding_Climate_Change.pdf --rebuild \
    --query "what is the greenhouse effect?"
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--path` | `""` | Path to PDF file (required for first run) |
| `--query` | `"what is the main cause of climate change?"` | Question to answer |
| `--n_retrieved` | `10` | Number of chunks to retrieve from FAISS |
| `--chunk_size` | `1000` | Chunk size for text splitting |
| `--chunk_overlap` | `200` | Overlap between chunks |
| `--rebuild` | `false` | Force rebuild persisted indexes |

## Project Structure

```
├── src/
│   ├── ingestion.py      # GraphRAG class (ingestion + query pipeline)
│   ├── util.py            # FAISS, KnowledgeGraph, traversal, visualization
│   └── __init__.py
├── data/                  # PDF documents
├── indexes/               # Persisted FAISS index + knowledge graph (auto-generated)
│   ├── faiss/
│   └── knowledge_graph.pkl
├── graph_traversal.png    # Auto-generated visualization
├── .env                   # API keys
└── readme.md
```

## Tech Stack

- **LLM:** Llama 3.3 70B via [Groq](https://groq.com/) (free tier)
- **Embeddings:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (HuggingFace, local)
- **Reranker:** [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (cross-encoder, local)
- **Vector Store:** FAISS (persisted to disk)
- **Knowledge Graph:** NetworkX + spaCy NER + LLM concept extraction
- **Chunking:** LangChain SemanticChunker (percentile-based breakpoints)
- **Framework:** LangChain

## Industry Practices Followed

1. **Offline ingestion / Online querying** — indexes built once, loaded from disk for queries
2. **Persisted vector store + knowledge graph** — no rebuilding on every run
3. **Query rewriting before retrieval** — improved query drives FAISS search
4. **Two-stage retrieval** — fast approximate (FAISS) → precise reranking (cross-encoder)
5. **Graph-expanded context** — Dijkstra traversal discovers conceptually connected chunks beyond vector similarity
6. **Single LLM call for generation** — predictable latency, no iterative answer-checking