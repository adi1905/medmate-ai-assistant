import os
from contextlib import asynccontextmanager
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from medmate.graph.main_graph import build_main_graph, shutdown_main_graph
from medmate.memory.repo import MemoryRepo

DSN = os.environ["MEDMATE_PG_DSN"]

@asynccontextmanager
async def lifespan_resources():
    pool = AsyncConnectionPool(
        conninfo=DSN, max_size=20, open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup() 
        store = AsyncPostgresStore(pool, index={
            "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
            "dims": 1536, "fields": ["text"], "distance_type": "cosine"})
        await store.setup() 
        repo = MemoryRepo(store)
        graph = await build_main_graph(checkpointer=checkpointer, repo=repo)
        try:
            yield graph, repo
        finally:
            await shutdown_main_graph(graph)
    finally:
        await pool.close()

def cfg(user_id: str, session_id: str) -> dict:
    # server-side only; never accept a thread_id from the client
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

async def send(graph, user, session_id, text):
    c = cfg(user.id, session_id)
    if (await graph.aget_state(c)).next:
        raise RuntimeError("Resolve the pending confirmation first.")
    return await graph.ainvoke(
        {"messages": [HumanMessage(text)], "user_id": user.id, "role": user.role}, c)

async def resolve(graph, user, session_id, decision):
    return await graph.ainvoke(Command(resume=decision), cfg(user.id, session_id))