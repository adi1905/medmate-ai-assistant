import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ..memory.hybrid_store import HybridMemoryStore
 
 
 
from config.settings import *
from .routers.auth.route import router as auth_router
from .routers.chat.route import router as chat_router
 
from .core.common_schemas import HealthResponse
 
from ..graph.main_graph import build_main_graph, shutdown_main_graph
 
logger = logging.getLogger("medmate.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(SQLITE_CHECKPOINTER_PATH)
    checkpointer = None
    qdrant = None
    store = None

    try:
        checkpointer = await checkpointer_cm.__aenter__()

        qdrant = QdrantClient(path=MEMORY_QDRANT_PATH)
        store = HybridMemoryStore(qdrant, SQLITE_EPISODIC_PATH)
        await store.setup()


        logger.info("Starting MedMate graph (this opens the lab-reference MCP client session)...")
        app.state.graph = await build_main_graph(checkpointer=checkpointer, store=store)
        app.state._checkpointer_cm = checkpointer_cm
        app.state._store = store
        app.state._qdrant = qdrant

        logger.info("Medmate graph ready")
        yield
    finally:
        logger.info("Shutting down MedMate graph...")
        if getattr(app.state, "graph", None) is not None:
            await shutdown_main_graph(app.state.graph)
 
        if checkpointer is not None:
            await checkpointer_cm.__aexit__(None, None, None)
 
        if store is not None:
            await store.close()
 
        if qdrant is not None:
            qdrant.close()
 
        logger.info("Shutdown complete.")
 
 
app = FastAPI(title="Medmate API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)
 
app.include_router(auth_router)
app.include_router(chat_router)
 
 
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))