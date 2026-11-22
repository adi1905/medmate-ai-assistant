from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=MODEL, temperature=0)

class SQLGeneration(BaseModel):
    sql: str = Field(..., description="A single valid SQLite SELECT statement")
    explanation: str = Field(..., description="One sentence explaining what the query does")
    tables_used: list[str] = Field(..., description="List of table names referenced in the query")
    answerable: bool = Field(..., description="False if the question cannot be answered from this schema")

structured_llm = llm.with_structured_output(SQLGeneration)

SYSTEM_PROMPT = """You are a SQL generation engine for a clinic analytics system.
You translate a staff member's natural-language question into a single SQLite SELECT query.

STRICT RULES:
- Only ever generate a single SELECT statement. Never INSERT, UPDATE, DELETE, DROP, ALTER,
  ATTACH, PRAGMA, CREATE, or any statement other than SELECT.
- Never generate multiple statements separated by semicolons.
- Only reference tables and columns that exist in the schema below. Do not invent columns.
- Use the EXACT enum values given in the schema (e.g. status values) — never guess or
  paraphrase them (e.g. use 'cancelled' not 'canceled', 'no_show' not 'no-show').
- Prefer explicit column names over SELECT *.
- Use JOINs via the documented foreign keys when a question spans multiple tables.
- Use standard SQLite date functions (date(), strftime()) for date bucketing/filtering.
- If the question cannot be answered from this schema, return a SELECT that returns zero
  rows from a valid table (e.g. SELECT patient_id FROM patients WHERE 1=0) rather than
  inventing data or querying a nonexistent table.
- Do not add a LIMIT unless the question specifically asks for a top-N result — the caller
  applies its own row cap afterward.

SCHEMA:
{schema}
"""

RETRY_ADDENDUM = """

Your previous attempt was rejected by the SQL safety validator for this reason:
{prior_error}

Generate a corrected query that fixes this issue while still answering the original question.
"""

def generate_sql(question: str, schema: str, prior_error: str | None = None) -> SQLGeneration:
    """
    Generates a single validated-format SQL SELECT statement for `question`,
    grounded in `schema` (from analytics_repo.get_schema_snapshot()).

    If `prior_error` is provided (from a failed sql_guard validation), it is
    appended to the prompt so the model can self-correct on retry.
    """
    system_prompt = SYSTEM_PROMPT.format(schema=schema)
    if prior_error:
        system_prompt += RETRY_ADDENDUM.format(prior_error=prior_error)

    result: SQLGeneration = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])
    return result