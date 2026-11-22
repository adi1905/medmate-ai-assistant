from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

MEMORY_PATH = r"C:\Users\adity\OneDrive\Documents\MedMate Clinic Copilot\data\memory_qdrant_store"

client = QdrantClient(path=MEMORY_PATH)

collection_name = "semantic_facts"

points, _ = client.scroll(
    collection_name=collection_name,
    limit=100,
    with_payload=True,
    with_vectors=False,
)

print(f"Found {len(points)} points\n")

for i, point in enumerate(points, 1):
    print("=" * 80)
    print(f"Point {i}")
    print("ID:", point.id)
    print("Payload:")
    print(point.payload)


point_id = "c58d6ce1-7736-4168-ac66-da6a25a15dd8"  # or an integer if your IDs are integers

client.delete(
    collection_name=collection_name,
    points_selector=PointIdsList(
        points=[point_id]
    ),
)

print("Point deleted.")