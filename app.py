# File: circlemind/app.py

import os
import datetime
import shutil
import uuid
import json
import logging
import sys
import asyncio
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("circlemind")

import fitz  # PyMuPDF for PDF extraction
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# fast-graphrag
from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import DefaultLLMService, DefaultEmbeddingService

# Ollama integration
from ollama_local import OllamaLLMService, OllamaEmbeddingService

# =============== FastAPI Setup ===============
app = FastAPI(
    title="Nationmind API", 
    description="Backend API for Nationmind using fast-graphrag",
    debug=True  # Enable debug mode
)

# Configure CORS if your React app is on a different domain:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== Data/Storage Setup ===============
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Path to the frontend build directory
FRONTEND_DIR = Path("./dist")  # Change to dist if that's where your built files are
INDEX_HTML = FRONTEND_DIR / "index.html"  # Update the index.html path

# Keep references to loaded GraphRAG instances in memory
graphs_cache = {}

# =============== Pydantic Models ===============
class GraphConfig(BaseModel):
    name: str
    domain: str
    entity_types: List[str]
    example_queries: Optional[List[str]] = []

class CreateGraphResponse(BaseModel):
    graph_id: str
    message: str

class GraphInfo(BaseModel):
    id: str
    name: str
    domain: str
    entity_types: List[str]
    example_queries: List[str]
    node_count: int
    edge_count: int
    chunk_count: int
    created_at: str
    last_updated: str

class DataUploadText(BaseModel):
    content: str = ""
    filename: Optional[str] = None

class GraphQuery(BaseModel):
    query: str
    with_references: bool = False

# =============== Helper Functions ===============

def get_graph_dir(graph_id: str) -> Path:
    """Return the folder path for a given graph_id."""
    return DATA_DIR / graph_id

def get_metadata_path(graph_id: str) -> Path:
    return get_graph_dir(graph_id) / "metadata.json"

def load_metadata(graph_id: str) -> dict:
    p = get_metadata_path(graph_id)
    if not p.exists():
        raise HTTPException(404, detail=f"Graph {graph_id} not found or missing metadata.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata(graph_id: str, meta: dict):
    p = get_metadata_path(graph_id)
    with p.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

async def ensure_graph_loaded(graph: GraphRAG):
    """Ensure the graph has loaded all its data from disk."""
    await graph.state_manager.query_start()
    try:
        # Try various loading methods
        if hasattr(graph.state_manager, "_load_all_storages"):
            await graph.state_manager._load_all_storages()
        # Add other loading methods as in the reload endpoint
    finally:
        await graph.state_manager.query_done()

def load_graphrag(graph_id: str) -> GraphRAG:
    """Load a GraphRAG instance from disk, or create a new one if it doesn't exist in memory."""
    if graph_id in graphs_cache:
        return graphs_cache[graph_id]

    g_dir = get_graph_dir(graph_id)
    if not g_dir.exists():
        raise HTTPException(404, detail=f"Graph {graph_id} not found.")

    meta = load_metadata(graph_id)
    
    # Check if we should use Ollama (local) or OpenAI (cloud)
    use_ollama = os.environ.get("USE_OLLAMA", "true").lower() in ("true", "1", "yes")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if use_ollama:
        logger.info("Using local Ollama models for LLM and embeddings")
        llm_service = OllamaLLMService(model="mistral-small")
        embedding_service = OllamaEmbeddingService(model="mxbai-embed-large")
    else:
        # Log a warning if the API key is missing
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. API calls will likely fail.")
        logger.info("Using OpenAI models for LLM and embeddings")
        llm_service = DefaultLLMService(api_key=openai_api_key)
        embedding_service = DefaultEmbeddingService(api_key=openai_api_key)

    gr = GraphRAG(
        working_dir=str(g_dir),
        domain=meta["domain"],
        entity_types=meta["entity_types"],
        example_queries="\n".join(meta.get("example_queries", [])),
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
    )

    # Cache for future use
    graphs_cache[graph_id] = gr
    
    # Schedule an async task to ensure data is loaded
    asyncio.create_task(ensure_graph_loaded(gr))
    
    return gr

async def update_node_edge_chunk_counts(graph_id: str, graph: GraphRAG):
    """Use the GraphRAG's state_manager to get real node/edge/chunk counts."""
    # We can't directly check the state, so we'll use a try/except approach
    try:
        logger.info("Starting count update process")
        
        # First try to get counts directly from the in-memory objects
        logger.info("Attempting to get counts from in-memory objects")
        ent_count = len(graph._entity_storage._entities) if hasattr(graph, "_entity_storage") and hasattr(graph._entity_storage, "_entities") else 0
        rel_count = len(graph._relation_storage._relations) if hasattr(graph, "_relation_storage") and hasattr(graph._relation_storage, "_relations") else 0
        chk_count = len(graph._chunk_storage._chunks) if hasattr(graph, "_chunk_storage") and hasattr(graph._chunk_storage, "_chunks") else 0
        
        logger.info(f"In-memory counts: {ent_count} entities, {rel_count} relations, {chk_count} chunks")
        
        # If we couldn't get counts from memory, try using the state manager
        if ent_count == 0 and rel_count == 0 and chk_count == 0:
            logger.info("In-memory counts are zero, trying state manager")
            
            # Switch to query mode to ensure we can access the data
            await graph.state_manager.query_start()
            try:
                ent_count = await graph.state_manager.get_num_entities()
                rel_count = await graph.state_manager.get_num_relations()
                chk_count = await graph.state_manager.get_num_chunks()
                logger.info(f"State manager counts: {ent_count} entities, {rel_count} relations, {chk_count} chunks")
            finally:
                await graph.state_manager.query_done()
        
        # Update metadata
        meta = load_metadata(graph_id)
        meta["node_count"] = ent_count
        meta["edge_count"] = rel_count
        meta["chunk_count"] = chk_count
        meta["last_updated"] = str(datetime.datetime.now())
        save_metadata(graph_id, meta)
        logger.info(f"Updated metadata with counts: {ent_count} entities, {rel_count} relations, {chk_count} chunks")
        
        return ent_count, rel_count, chk_count
    except Exception as e:
        logger.error(f"Error updating counts: {str(e)}", exc_info=True)
        # If we fail, just return zeros
        return 0, 0, 0

# =============== FastAPI Routes ===============

@app.get("/")
def root():
    # Instead of returning API message, serve the frontend
    return FileResponse(INDEX_HTML)

@app.post("/graphs", response_model=CreateGraphResponse)
async def create_graph(cfg: GraphConfig):
    """Create a new GraphRAG instance on disk and store metadata."""
    graph_id = str(uuid.uuid4())
    g_dir = get_graph_dir(graph_id)
    g_dir.mkdir(parents=True, exist_ok=False)

    now_str = str(datetime.datetime.now())

    metadata = {
        "id": graph_id,
        "name": cfg.name,
        "domain": cfg.domain,
        "entity_types": cfg.entity_types,
        "example_queries": cfg.example_queries,
        "node_count": 0,
        "edge_count": 0,
        "chunk_count": 0,
        "created_at": now_str,
        "last_updated": now_str
    }

    save_metadata(graph_id, metadata)

    # Initialize the GraphRAG
    _ = load_graphrag(graph_id)

    return CreateGraphResponse(
        graph_id=graph_id,
        message=f"Graph '{cfg.name}' created successfully."
    )

@app.get("/graphs", response_model=List[GraphInfo])
async def list_graphs():
    """List all available graphs."""
    result = []
    for g_dir in DATA_DIR.iterdir():
        if g_dir.is_dir():
            try:
                meta = load_metadata(g_dir.name)
                result.append(GraphInfo(
                    id=meta["id"],
                    name=meta["name"],
                    domain=meta["domain"],
                    entity_types=meta["entity_types"],
                    example_queries=meta.get("example_queries", []),
                    node_count=meta["node_count"],
                    edge_count=meta["edge_count"],
                    chunk_count=meta.get("chunk_count", 0),
                    created_at=meta["created_at"],
                    last_updated=meta["last_updated"]
                ))
            except Exception:
                # Skip any directories that don't have valid metadata
                continue
    return result

@app.get("/graphs/{graph_id}", response_model=GraphInfo)
async def get_graph_info(graph_id: str):
    """Get detailed info about a specific graph."""
    meta = load_metadata(graph_id)

    # Optionally refresh counts if the user wants up-to-date info
    # We do that by loading the graph from cache or disk.
    g = load_graphrag(graph_id)
    await update_node_edge_chunk_counts(graph_id, g)
    meta = load_metadata(graph_id)  # re-load updated meta

    return GraphInfo(
        id=meta["id"],
        name=meta["name"],
        domain=meta["domain"],
        entity_types=meta["entity_types"],
        example_queries=meta.get("example_queries", []),
        node_count=meta["node_count"],
        edge_count=meta["edge_count"],
        chunk_count=meta.get("chunk_count", 0),
        created_at=meta["created_at"],
        last_updated=meta["last_updated"]
    )

@app.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a graph and remove from disk/cache."""
    g_dir = get_graph_dir(graph_id)
    if not g_dir.exists():
        raise HTTPException(404, detail="Graph not found.")

    if graph_id in graphs_cache:
        del graphs_cache[graph_id]

    shutil.rmtree(g_dir)
    return {"message": f"Graph {graph_id} deleted successfully."}

@app.post("/graphs/{graph_id}/upload-text")
async def upload_text_data(graph_id: str, data: DataUploadText):
    """Upload raw text to the graph (JSON body)."""
    logger.info(f"Processing text upload for graph {graph_id}")
    graph = load_graphrag(graph_id)

    if not data.content.strip():
        raise HTTPException(400, detail="No text content provided.")

    # Log content summary
    content_preview = data.content[:100] + "..." if len(data.content) > 100 else data.content
    logger.info(f"Text content preview: {content_preview}")
    logger.info(f"Content length: {len(data.content)} characters")

    # Insert text
    try:
        logger.info("Starting text insertion process")
        await graph.state_manager.insert_start()
        meta_dict = {"filename": data.filename or "raw_text"}
        
        logger.info("Calling graph.async_insert")
        ent_count, rel_count, chunk_count = await graph.async_insert(
            content=data.content, 
            metadata=meta_dict
        )
        logger.info(f"Insertion complete. Extracted: {ent_count} entities, {rel_count} relations, {chunk_count} chunks")
        
        # Force a commit to disk before finalizing
        logger.info("Forcing commit to disk")
        await graph.state_manager._commit_all_storages()
        
        await graph.state_manager.insert_done()
        logger.info("Insertion process finalized")

        # Update counts in metadata
        logger.info("Updating node/edge/chunk counts")
        await update_node_edge_chunk_counts(graph_id, graph)
        
        # Log the updated counts from metadata
        meta = load_metadata(graph_id)
        logger.info(f"Updated metadata counts: {meta['node_count']} nodes, {meta['edge_count']} edges, {meta['chunk_count']} chunks")

        # Log the graph directory contents
        g_dir = get_graph_dir(graph_id)
        logger.info(f"Files in graph directory: {[f.name for f in g_dir.iterdir()]}")

        return {"message": "Text data ingested successfully."}
    except Exception as e:
        logger.error(f"Error during text insertion: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/graphs/{graph_id}/upload-file")
async def upload_file_data(graph_id: str, file: UploadFile = File(...)):
    """Upload a file (PDF or TXT) to be ingested."""
    logger.info(f"Processing file upload for graph {graph_id}: {file.filename}")
    graph = load_graphrag(graph_id)

    original_name = file.filename
    extension = (original_name or "").lower().split(".")[-1]
    logger.info(f"File extension: {extension}")

    # Read the file into memory
    content_bytes = await file.read()
    logger.info(f"Read {len(content_bytes)} bytes from file")

    if extension == "pdf":
        # Extract text from PDF
        try:
            logger.info("Processing PDF file")
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            text_parts = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                text_parts.append(page_text)
                logger.info(f"Extracted {len(page_text)} characters from page {page_num+1}")
            doc.close()
            text_data = "\n".join(text_parts)
            logger.info(f"Total extracted text: {len(text_data)} characters from {len(text_parts)} pages")
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}", exc_info=True)
            raise HTTPException(400, detail=f"Failed to process PDF: {str(e)}")
    elif extension in ("txt", "md"):
        logger.info(f"Processing {extension} file")
        text_data = content_bytes.decode("utf-8", errors="replace")
        logger.info(f"Decoded {len(text_data)} characters of text")
    else:
        logger.warning(f"Unsupported file extension: {extension}")
        raise HTTPException(400, detail=f"Unsupported file extension: {extension}")

    if not text_data.strip():
        logger.warning("File is empty or contains no extractable text")
        raise HTTPException(400, detail="File is empty or unreadable.")

    # Log content preview
    content_preview = text_data[:100] + "..." if len(text_data) > 100 else text_data
    logger.info(f"Content preview: {content_preview}")

    try:
        logger.info("Starting file content insertion")
        await graph.state_manager.insert_start()
        
        # Insert with some metadata
        meta_dict = {"filename": original_name}
        logger.info("Calling graph.async_insert")
        ent_count, rel_count, chunk_count = await graph.async_insert(
            content=text_data, 
            metadata=meta_dict
        )
        logger.info(f"Insertion complete. Extracted: {ent_count} entities, {rel_count} relations, {chunk_count} chunks")
        
        # Try to force a commit before finalizing
        try:
            logger.info("Attempting to force commit to disk")
            if hasattr(graph.state_manager, "_commit_all_storages"):
                await graph.state_manager._commit_all_storages()
            elif hasattr(graph, "_commit"):
                await graph._commit()
            else:
                logger.warning("No commit method found, relying on insert_done to commit")
        except Exception as commit_err:
            logger.warning(f"Error during forced commit: {str(commit_err)}")
        
        # Finalize the insertion
        await graph.state_manager.insert_done()
        logger.info("Insertion process finalized")

        # Update metadata with the counts we got from the insertion
        meta = load_metadata(graph_id)
        meta["node_count"] = ent_count
        meta["edge_count"] = rel_count
        meta["chunk_count"] = chunk_count
        meta["last_updated"] = str(datetime.datetime.now())
        save_metadata(graph_id, meta)
        logger.info(f"Updated metadata with insertion counts: {ent_count} entities, {rel_count} relations, {chunk_count} chunks")

        # Log the graph directory contents
        g_dir = get_graph_dir(graph_id)
        logger.info(f"Files in graph directory: {[f.name for f in g_dir.iterdir()]}")

        return {"message": f"File '{original_name}' ingested successfully."}
    except Exception as e:
        logger.error(f"Error during file insertion: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/graphs/{graph_id}/query")
async def query_graph(graph_id: str, q: GraphQuery):
    """Ask a query of the specified graph using GraphRAG."""
    logger.info(f"Processing query for graph {graph_id}: {q.query}")
    graph = load_graphrag(graph_id)

    try:
        logger.info("Starting query process")
        await graph.state_manager.query_start()
        param = QueryParam(with_references=q.with_references, only_context=False)
        
        logger.info("Executing async_query")
        answer = await graph.async_query(q.query, params=param)
        logger.info(f"Query complete. Response length: {len(answer.response)}")
        
        await graph.state_manager.query_done()
        logger.info("Query process finalized")
        
        # Log response preview
        response_preview = answer.response[:100] + "..." if len(answer.response) > 100 else answer.response
        logger.info(f"Response preview: {response_preview}")
        
        # Process context data safely
        context_data = None
        if q.with_references and answer.context:
            try:
                # Check if to_dict method exists
                if hasattr(answer.context, 'to_dict'):
                    context_data = answer.context.to_dict()
                # If not, try to extract references directly
                elif hasattr(answer.context, 'references'):
                    references = []
                    for ref in answer.context.references:
                        ref_data = {
                            "id": str(ref.id) if hasattr(ref, 'id') else str(uuid.uuid4()),
                            "text": ref.text if hasattr(ref, 'text') else "",
                        }
                        if hasattr(ref, 'metadata'):
                            ref_data["metadata"] = ref.metadata
                        references.append(ref_data)
                    context_data = {"references": references}
                # If all else fails, convert to string
                else:
                    context_data = {"raw_context": str(answer.context)}
                
                logger.info(f"Context references: {len(context_data.get('references', [])) if isinstance(context_data, dict) and 'references' in context_data else 0}")
            except Exception as e:
                logger.error(f"Error processing context: {str(e)}", exc_info=True)
                context_data = {"error": "Failed to process context data"}
        
        return {
            "query": q.query,
            "response": answer.response,
            "context": context_data
        }
    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        await graph.state_manager.query_done()
        raise HTTPException(500, detail=str(e))

@app.patch("/graphs/{graph_id}")
async def update_graph_metadata(graph_id: str, cfg: GraphConfig):
    """Update domain, entity_types, or example_queries for an existing graph."""
    meta = load_metadata(graph_id)

    # Overwrite any fields provided
    meta["name"] = cfg.name
    meta["domain"] = cfg.domain
    meta["entity_types"] = cfg.entity_types
    meta["example_queries"] = cfg.example_queries or []
    meta["last_updated"] = str(datetime.datetime.now())

    save_metadata(graph_id, meta)

    # Clear from cache so a fresh instance is created next time
    if graph_id in graphs_cache:
        del graphs_cache[graph_id]

    return {"message": f"Graph {graph_id} updated successfully."}

@app.post("/graphs/{graph_id}/refresh-counts")
async def refresh_counts(graph_id: str):
    """Force a refresh of the node/edge/chunk counts."""
    logger.info(f"Manually refreshing counts for graph {graph_id}")
    graph = load_graphrag(graph_id)
    
    try:
        await update_node_edge_chunk_counts(graph_id, graph)
        meta = load_metadata(graph_id)
        return {
            "node_count": meta["node_count"],
            "edge_count": meta["edge_count"],
            "chunk_count": meta["chunk_count"]
        }
    except Exception as e:
        logger.error(f"Error refreshing counts: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.get("/graphs/{graph_id}/counts")
async def get_counts(graph_id: str):
    """Get the current node/edge/chunk counts for a graph."""
    meta = load_metadata(graph_id)
    return {
        "node_count": meta.get("node_count", 0),
        "edge_count": meta.get("edge_count", 0),
        "chunk_count": meta.get("chunk_count", 0)
    }

# ---------- NEW ENDPOINT: Explorer data (Entities / Relationships / Chunks) -----------
@app.get("/graphs/{graph_id}/explorer")
async def get_graph_explorer_data(graph_id: str):
    """
    Return a list of entities, relationships, and chunk IDs so the frontend can render them.
    """
    logger.info(f"Fetching explorer data for graph {graph_id}")
    graph = load_graphrag(graph_id)
    
    # First, let's check what the metadata says
    meta = load_metadata(graph_id)
    logger.info(f"Metadata shows: {meta.get('node_count', 0)} nodes, {meta.get('edge_count', 0)} edges, {meta.get('chunk_count', 0)} chunks")
    
    # Switch to query mode to read from storages
    await graph.state_manager.query_start()

    # We'll attempt to retrieve all entity/edge info from the underlying storage.
    entities = []
    relationships = []
    chunks = set()

    try:
        logger.info("Retrieving entity and relationship data")
        node_count = await graph.state_manager.get_num_entities()
        edge_count = await graph.state_manager.get_num_relations()
        chunk_count = await graph.state_manager.get_num_chunks()
        
        # Convert NumPy types to Python native types
        node_count = int(node_count) if hasattr(node_count, "item") else node_count
        edge_count = int(edge_count) if hasattr(edge_count, "item") else edge_count
        chunk_count = int(chunk_count) if hasattr(chunk_count, "item") else chunk_count
        
        logger.info(f"Actual storage contains: {node_count} entities, {edge_count} relationships, {chunk_count} chunks")

        # Create a mapping of node IDs to node data
        node_map = {}
        
        # For each node index, get the node
        for i in range(node_count):
            try:
                node = await graph.state_manager.graph_storage.get_node_by_index(i)
                if node:
                    # Convert any NumPy types to Python native types
                    node_id = str(node.id) if hasattr(node, "id") else str(i)
                    node_name = str(node.name) if hasattr(node, "name") else f"Node-{i}"
                    node_type = str(node.type) if hasattr(node, "type") else "unknown"
                    node_desc = str(node.description) if hasattr(node, "description") else ""
                    
                    entity_data = {
                        "index": int(i),
                        "id": node_id,
                        "name": node_name,
                        "type": node_type,
                        "description": node_desc
                    }
                    
                    entities.append(entity_data)
                    
                    # Store in map for edge lookups
                    node_map[str(i)] = {
                        "name": node_name,
                        "type": node_type,
                        "description": node_desc
                    }
                    node_map[node_id] = {
                        "name": node_name,
                        "type": node_type,
                        "description": node_desc
                    }
                else:
                    logger.warning(f"Node at index {i} returned None")
            except Exception as e:
                logger.error(f"Error retrieving node at index {i}: {str(e)}")

        # For each edge index, get the edge
        for i in range(edge_count):
            try:
                edge = await graph.state_manager.graph_storage.get_edge_by_index(i)
                if edge:
                    # Convert source and target to strings to avoid NumPy type issues
                    source_id = str(edge.source) if hasattr(edge, "source") else ""
                    target_id = str(edge.target) if hasattr(edge, "target") else ""
                    
                    # Get source and target info
                    source_info = node_map.get(source_id, {"name": f"Node-{source_id}", "type": "unknown"})
                    target_info = node_map.get(target_id, {"name": f"Node-{target_id}", "type": "unknown"})
                    
                    # Convert edge description and chunks to native Python types
                    edge_desc = str(edge.description) if hasattr(edge, "description") else ""
                    edge_chunks = []
                    
                    if hasattr(edge, "chunks") and edge.chunks:
                        for c in edge.chunks:
                            chunk_id = str(c)
                            edge_chunks.append(chunk_id)
                            chunks.add(chunk_id)
                    
                    relationship_data = {
                        "index": int(i),
                        "source": source_info["name"],
                        "target": target_info["name"],
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_type": source_info.get("type", "unknown"),
                        "target_type": target_info.get("type", "unknown"),
                        "description": edge_desc,
                        "predicate": edge_desc,  # Adding predicate for clarity in UI
                        "chunks": edge_chunks
                    }
                    
                    relationships.append(relationship_data)
                else:
                    logger.warning(f"Edge at index {i} returned None")
            except Exception as e:
                logger.error(f"Error retrieving edge at index {i}: {str(e)}")
        
        # Convert chunks set to a list of strings
        chunk_list = [str(c) for c in chunks]
        
        logger.info(f"Processed {len(entities)} entities, {len(relationships)} relationships, and {len(chunk_list)} chunks")
    except Exception as e:
        logger.error(f"Error retrieving explorer data: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Failed to retrieve graph data: {str(e)}")
    finally:
        await graph.state_manager.query_done()
        logger.info("Query mode ended")

    return {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunk_list
    }

@app.post("/graphs/{graph_id}/fix-metadata")
async def fix_metadata(graph_id: str):
    """Fix incorrect metadata by updating it to match actual storage counts."""
    logger.info(f"Fixing metadata for graph {graph_id}")
    graph = load_graphrag(graph_id)
    
    await graph.state_manager.query_start()
    try:
        node_count = await graph.state_manager.get_num_entities()
        edge_count = await graph.state_manager.get_num_relations()
        chunk_count = await graph.state_manager.get_num_chunks()
        
        meta = load_metadata(graph_id)
        meta["node_count"] = node_count
        meta["edge_count"] = edge_count
        meta["chunk_count"] = chunk_count
        meta["last_updated"] = str(datetime.datetime.now())
        save_metadata(graph_id, meta)
        
        logger.info(f"Updated metadata to match actual counts: {node_count} nodes, {edge_count} edges, {chunk_count} chunks")
        return {"message": "Metadata fixed successfully", "counts": {"nodes": node_count, "edges": edge_count, "chunks": chunk_count}}
    finally:
        await graph.state_manager.query_done()

@app.post("/graphs/{graph_id}/reload-storage")
async def reload_graph_storage(graph_id: str):
    """Force reload of graph storage from disk files."""
    logger.info(f"Forcing reload of graph storage for {graph_id}")
    
    # First, remove from cache if present
    if graph_id in graphs_cache:
        logger.info("Removing graph from cache")
        del graphs_cache[graph_id]
    
    # Load graph with fresh instance
    graph = load_graphrag(graph_id)
    
    # Force initialization of all storages
    logger.info("Initializing state manager and storages")
    await graph.state_manager.query_start()
    
    try:
        # Try to force load from disk
        if hasattr(graph.state_manager, "_load_all_storages"):
            logger.info("Calling _load_all_storages")
            await graph.state_manager._load_all_storages()
        elif hasattr(graph.state_manager, "reload"):
            logger.info("Calling reload")
            await graph.state_manager.reload()
        else:
            logger.warning("No direct reload method found, trying individual storages")
            
            # Try to reload individual storages
            if hasattr(graph.state_manager, "graph_storage") and hasattr(graph.state_manager.graph_storage, "reload"):
                logger.info("Reloading graph storage")
                await graph.state_manager.graph_storage.reload()
            
            if hasattr(graph.state_manager, "chunk_storage") and hasattr(graph.state_manager.chunk_storage, "reload"):
                logger.info("Reloading chunk storage")
                await graph.state_manager.chunk_storage.reload()
            
            if hasattr(graph.state_manager, "embedding_storage") and hasattr(graph.state_manager.embedding_storage, "reload"):
                logger.info("Reloading embedding storage")
                await graph.state_manager.embedding_storage.reload()
        
        # Get counts after reload
        node_count = await graph.state_manager.get_num_entities()
        edge_count = await graph.state_manager.get_num_relations()
        chunk_count = await graph.state_manager.get_num_chunks()
        
        logger.info(f"After reload: {node_count} entities, {edge_count} relationships, {chunk_count} chunks")
        
        # Update metadata to match actual counts
        meta = load_metadata(graph_id)
        meta["node_count"] = node_count
        meta["edge_count"] = edge_count
        meta["chunk_count"] = chunk_count
        meta["last_updated"] = str(datetime.datetime.now())
        save_metadata(graph_id, meta)
        
        return {
            "message": "Graph storage reloaded successfully",
            "counts": {
                "nodes": node_count,
                "edges": edge_count,
                "chunks": chunk_count
            }
        }
    except Exception as e:
        logger.error(f"Error reloading graph storage: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Failed to reload graph storage: {str(e)}")
    finally:
        await graph.state_manager.query_done()

@app.post("/graphs/{graph_id}/diagnose")
async def diagnose_graph_storage(graph_id: str):
    """Diagnose graph storage issues and attempt various fixes."""
    logger.info(f"Running diagnostic on graph {graph_id}")
    
    # Get the graph directory
    g_dir = get_graph_dir(graph_id)
    
    # Check what files exist
    files = list(g_dir.glob("*"))
    file_info = [{"name": f.name, "size": f.stat().st_size, "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()} for f in files]
    logger.info(f"Found {len(files)} files in graph directory: {[f.name for f in files]}")
    
    # Load metadata
    meta = load_metadata(graph_id)
    logger.info(f"Metadata: {meta}")
    
    # Remove from cache if present
    if graph_id in graphs_cache:
        logger.info("Removing graph from cache")
        del graphs_cache[graph_id]
    
    # Create a fresh GraphRAG instance with explicit settings
    logger.info("Creating fresh GraphRAG instance")
    use_ollama = os.environ.get("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if use_ollama:
        logger.info("Using local Ollama models for LLM and embeddings")
        llm_service = OllamaLLMService(model="mistral-small")
        embedding_service = OllamaEmbeddingService(model="mxbai-embed-large")
    else:
        # Log a warning if the API key is missing
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. API calls will likely fail.")
        logger.info("Using OpenAI models for LLM and embeddings")
        llm_service = DefaultLLMService(api_key=openai_api_key)
        embedding_service = DefaultEmbeddingService(api_key=openai_api_key)

    gr = GraphRAG(
        working_dir=str(g_dir),
        domain=meta["domain"],
        entity_types=meta["entity_types"],
        example_queries="\n".join(meta.get("example_queries", [])),
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
    )
    
    # Try various loading approaches
    results = {}
    
    # 1. Try direct loading
    logger.info("Approach 1: Direct loading")
    await gr.state_manager.query_start()
    try:
        node_count = await gr.state_manager.get_num_entities()
        edge_count = await gr.state_manager.get_num_relations()
        chunk_count = await gr.state_manager.get_num_chunks()
        results["direct_loading"] = {
            "node_count": node_count,
            "edge_count": edge_count,
            "chunk_count": chunk_count
        }
        logger.info(f"Direct loading: {node_count} nodes, {edge_count} edges, {chunk_count} chunks")
    finally:
        await gr.state_manager.query_done()
    
    # 2. Try with explicit load_all_storages
    logger.info("Approach 2: Explicit storage loading")
    await gr.state_manager.query_start()
    try:
        if hasattr(gr.state_manager, "_load_all_storages"):
            await gr.state_manager._load_all_storages()
        node_count = await gr.state_manager.get_num_entities()
        edge_count = await gr.state_manager.get_num_relations()
        chunk_count = await gr.state_manager.get_num_chunks()
        results["explicit_loading"] = {
            "node_count": node_count,
            "edge_count": edge_count,
            "chunk_count": chunk_count
        }
        logger.info(f"Explicit loading: {node_count} nodes, {edge_count} edges, {chunk_count} chunks")
    finally:
        await gr.state_manager.query_done()
    
    # 3. Try a query to see if it works
    logger.info("Approach 3: Testing query functionality")
    try:
        await gr.state_manager.query_start()
        param = QueryParam(with_references=True, only_context=False)
        answer = await gr.async_query("What is this graph about?", params=param)
        await gr.state_manager.query_done()
        
        results["query_test"] = {
            "response": answer.response[:100] + "..." if len(answer.response) > 100 else answer.response,
            "has_context": answer.context is not None,
            "context_type": str(type(answer.context))
        }
        logger.info(f"Query test: Response received, has context: {answer.context is not None}")
    except Exception as e:
        results["query_test"] = {"error": str(e)}
        logger.error(f"Query test failed: {str(e)}")
    
    # 4. Check if we need to rebuild the graph
    logger.info("Approach 4: Checking if rebuild is needed")
    needs_rebuild = all(count == 0 for approach in ["direct_loading", "explicit_loading"] 
                       for count in results.get(approach, {}).values() if isinstance(count, int))
    
    results["diagnosis"] = {
        "files_found": len(files),
        "metadata_node_count": meta.get("node_count", 0),
        "metadata_edge_count": meta.get("edge_count", 0),
        "metadata_chunk_count": meta.get("chunk_count", 0),
        "needs_rebuild": needs_rebuild,
        "query_works": "error" not in results.get("query_test", {})
    }
    
    # Update the cache with our new instance
    graphs_cache[graph_id] = gr
    
    return {
        "graph_id": graph_id,
        "file_info": file_info,
        "results": results,
        "recommendation": "Your graph appears to be working for queries but has no entities/relationships in storage. "
                         "This likely means entity extraction didn't work during ingestion. "
                         "Try re-uploading your content with more specific domain and entity_types settings."
                         if needs_rebuild and "error" not in results.get("query_test", {}) else
                         "Your graph appears to be completely broken. Try creating a new graph."
                         if needs_rebuild else
                         "Your graph appears to be working correctly."
    }

# =============== Serve Frontend ===============
# Mount the static files directory
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

# Serve the index.html for all non-API routes
@app.get("/{full_path:path}")
async def serve_frontend(request: Request, full_path: str):
    # If the path starts with /graphs, it's an API call
    if full_path.startswith("graphs"):
        # This is an API route, so raise a 404 to let the API handlers take over
        raise HTTPException(status_code=404, detail="Not found")
    
    # For all other paths, serve the index.html
    return FileResponse(INDEX_HTML)

# If you want to run via "python app.py" directly:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Enable auto-reloading
        log_level="info"
    )
