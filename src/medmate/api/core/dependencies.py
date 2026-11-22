from fastapi import Request


def get_graph(request: Request):
    """
    The compiled MedMate graph is built once in main.py's lifespan (it
    holds a live MCP client session and shouldn't be reconstructed per
    request) and stored on app.state. Routes pull it via this dependency
    instead of importing/building it themselves.
    """
    graph = request.app.state.graph
    if graph is None:
        # Only happens if build_main_graph(...) in main.py's lifespan is
        # still a TODO -- see main.py.
        raise RuntimeError("Graph not initialized -- wire up build_main_graph() in main.py's lifespan")
    return graph