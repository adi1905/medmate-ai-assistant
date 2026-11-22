import hashlib

from langchain_core.documents import Document


EXCLUDED_METADATA = {
    "raw_table",
    "table_json",
}


def generate_chunk_id(
    content: str,
    metadata: dict,
) -> str:

    key = (
        f"{metadata.get('source')}|"
        f"{metadata.get('page')}|"
        f"{content[:150]}"
    )

    return hashlib.md5(
        key.encode()
    ).hexdigest()


def copy_metadata(
    metadata: dict,
) -> dict:

    return {
        k: v
        for k, v in metadata.items()
        if k not in EXCLUDED_METADATA
    }