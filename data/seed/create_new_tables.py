from config.settings import DB_PATH, SCHEMA_PATH
import sqlite3

with open(SCHEMA_PATH, "r") as f:
    schema_sql = f.read()

conn = sqlite3.connect(DB_PATH)
conn.executescript(schema_sql)
cur = conn.cursor()