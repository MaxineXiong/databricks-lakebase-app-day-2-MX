# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Data -> Vector Embeddings (Lakebase) -> Vector Search

# COMMAND ----------

# Install python libraries
!pip install -r requirements.txt
!pip uninstall -y psycopg2 psycopg2-binary
!pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

%load_ext autoreload
%autoreload 2
# Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# To disable autoreload; run %autoreload 0

# COMMAND ----------

# Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Lakebase Setup

# COMMAND ----------

# Set configuration parameters
dbutils.widgets.text("weather_table_name", "weather_documents", "Destination table (raw weather)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")  # An embedding model from HuggingFace
dbutils.widgets.text("docs_fetch_limit", "50", "Max documents to fetch per location")
dbutils.widgets.text("chunk_size", "800", "Document content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Document content chunk overlap (chars)")

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
DOCS_FETCH_LIMIT = int(dbutils.widgets.get("docs_fetch_limit"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient
from lakebase import _lakebase_url

w = WorkspaceClient()

lakebase_url = _lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# Test Psycopg2 connection
import psycopg2
from lakebase import get_connection
import traceback

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using OAuth token authentication as user: {db_user}\n")

# Test psycopg3 connection with OAuth token
try:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables;")
            count = cursor.fetchall()[0]['count']
            print(f"✅ Connection successful!")
    print("\n✅ psycopg2 with OAuth authentication working correctly!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# Initialise weather tables in Lakebase
from lakebase import init_schema, run_query

query_results = run_query(
    """
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name IN ('weather_documents', 'weather_embeddings');
    """
)
count = query_results[0]['count']

if count < 2:
    init_schema()
    print(f"✅ Created tables {WEATHER_TABLE_NAME} and {EMBEDDINGS_TABLE_NAME}")
else:
    print(f"✅ Tables {WEATHER_TABLE_NAME} and {EMBEDDINGS_TABLE_NAME} already exist")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Weather Data and Load into Lakebase

# COMMAND ----------

# Fetch weather data for listed locations
import pandas as pd
from weather_client import WeatherClient

# Initialize weather client
client = WeatherClient()

# Read locations from CSV (columns: city, state, lat, lon)
locations_df = pd.read_csv('locations.csv')
print(f"Loaded {len(locations_df)} locations from locations.csv")

# Fetch weather data for each location
all_documents = []
for idx, row in locations_df.iterrows():
    city = row['city']
    state = row['state']
    lat = row['lat']
    lon = row['lon']
    
    try:
        # Pass city, state, lat, lon to populate all location fields
        docs = client.fetch_weather_data(
            city=city, 
            state=state, 
            lat=lat, 
            lon=lon,
            limit=DOCS_FETCH_LIMIT,
        )
        all_documents += docs
        print(f"✓ {city}, {state}: {len(docs)} documents")
    except Exception as e:
        print(f"✗ {city}, {state}: Error - {e}")

# Convert all documents to DataFrame
weather_df = pd.DataFrame(all_documents)

print(f"\n{'='*60}")
print(f"Total documents fetched: {len(weather_df)}")
print(f"Unique cities: {weather_df['city'].nunique()}")
print(f"Unique states: {weather_df['state'].nunique()}")
print(f"Source types: {weather_df['source_type'].value_counts().to_dict()}")
print(f"{'='*60}\n")

display(weather_df.head(10))

# COMMAND ----------

# Load weather data into Lakebase
import psycopg2
import json
from datetime import datetime
from lakebase import get_connection

with get_connection() as conn:
    with conn.cursor() as cursor:

        # Prepare data tuples for batch insert
        new_rows = []
        for _, row in weather_df.iterrows():
            # Convert payload dict to JSON string
            payload_json = json.dumps(row['payload'])
            
            # Parse timestamps to datetime objects
            effective_at = datetime.fromisoformat(row['effective_at'].replace('Z', '+00:00')) if row['effective_at'] else None
            synced_at = datetime.fromisoformat(row['synced_at'].replace('Z', '+00:00')) if row['synced_at'] else None
            
            # Combine city and state into location field
            location = f"{row['city']}, {row['state']}"
            
            new_rows.append(
                (
                    row['id'],
                    location,
                    row['source_type'],
                    row['headline'],
                    row['narrative_text'],
                    effective_at,
                    payload_json,
                    synced_at
                )
            )
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql_query = f"""
            INSERT INTO {WEATHER_TABLE_NAME} (
                id, location, source_type, headline, narrative_text,
                effective_at, payload, synced_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """
        
        # executemany in psycopg2 is much faster than individual INSERTs
        cursor.executemany(insert_sql_query, new_rows)
        conn.commit()

        print(f"✅ Successfully inserted {cursor.rowcount} new weather documents")
        print(f"   Total documents processed: {len(weather_df)}")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")

print(f"\n✅ Weather data loaded into Lakebase!")

# COMMAND ----------

# Query weather data from Lakebase
import pandas as pd
from lakebase import run_query

# Query with embedding_text computed
query = f"""
    SELECT 
        id,
        location,
        source_type,
        headline,
        TRIM(narrative_text) AS embedding_text,
        effective_at,
        synced_at
    FROM {WEATHER_TABLE_NAME}
    WHERE TRIM(narrative_text) IS NOT NULL
      AND TRIM(narrative_text) <> ''
"""

query_results = run_query(query)
weather_df = pd.DataFrame(query_results)

print(f"\nLoaded {len(weather_df)} weather documents from {WEATHER_TABLE_NAME}")
display(weather_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings on Chunked Narrative Text

# COMMAND ----------

# Chunk Narrative Text
import pandas as pd
import requests
import trafilatura

# Filter weather_df for documents with valid narrative text
content_df = weather_df.copy()

print(f"Chunking content from {len(content_df)} documents...")

# Chunk document content
out_doc_ids, out_chunk_indexes, out_chunk_texts = [], [], []

for idx, row in content_df.iterrows():
    document_id = row['id']
    text = row['embedding_text']

    # Split into overlapping chunks
    for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_doc_ids.append(document_id)
        out_chunk_indexes.append(str(chunk_index))
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(text):
            break
    
    # Progress update every 10 documents
    if ((idx + 1) % 10 == 0) or ((idx + 1) == len(content_df)):
        print(f"  Processed {idx + 1}/{len(content_df)} documents")

chunks_df = pd.DataFrame({
    "document_id": out_doc_ids,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Extracted {len(chunks_df)} content chunks from {len(content_df)} documents")
display(chunks_df.head())

# COMMAND ----------

# Compute embeddings on content chunks
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Model should already be loaded from earlier, but ensure cache is set
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
# Reuse the model if already loaded, otherwise load it
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute chunk embeddings in batches
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i : i + batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings += vectors.tolist()
    if ((i + batch_size) % 128 == 0) or ((i + batch_size) >= len(chunks_df)):
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame
chunk_embeddings_df = pd.DataFrame({
    "document_id": chunks_df["document_id"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert Chunk Embeddings into Lakebase

# COMMAND ----------

# Load chunk embeddings into Lakebase
import psycopg2
from datetime import datetime
from lakebase import get_connection

# Add id (document_id_chunk_index), model_name, and embedded_at columns
chunk_embeddings_df['id'] = chunk_embeddings_df['document_id'] + '_' + chunk_embeddings_df['chunk_index'].astype(str)
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['created_at'] = datetime.now()
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Prepare data tuples for batch insert
            # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
            new_rows = [
                (
                    row['id'],
                    row['document_id'],
                    int(row['chunk_index']),
                    row['chunk_text'],
                    '[' + ','.join(str(float(x)) for x in row['embedding']) + ']',
                    row['model_name'],
                    row['created_at']
                )
                for row in chunk_embeddings_rows
            ]
            
            # Batch insert with ON CONFLICT DO NOTHING for deduplication
            insert_sql_query = f"""
                INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                    id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
                ) VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """
            
            # executemany in psycopg2 is much faster than individual INSERTs
            cursor.executemany(insert_sql_query, new_rows)
            conn.commit()
            print(f"✅ Successfully inserted {cursor.rowcount} new chunk embeddings")
            print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
else:
    print("No chunk embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Backend Vector Search Query

# COMMAND ----------

# Test vector search with sample query
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
from lakebase import run_query

# Test query - find weather information about severe weather
test_query = "severe thunderstorms with heavy rain and flooding"

print(f"Testing vector search with query: '{test_query}'\n")

# Load model and compute query embedding
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

query_embedding = model.encode([test_query])[0].tolist()
embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'

# Query for top 10 most similar chunks using cosine distance
vector_search_query = f"""
    SELECT 
        e.id,
        w.location,
        w.headline,
        w.source_type,
        e.chunk_text,
        w.effective_at,
        1 - (e.embedding <=> %s::vector) AS similarity
    FROM {EMBEDDINGS_TABLE_NAME} e
    JOIN {WEATHER_TABLE_NAME} w ON e.document_id = w.id
    WHERE e.embedding IS NOT NULL
    ORDER BY similarity DESC
    LIMIT %s
"""

query_results = run_query(vector_search_query,(embedding_str, 10))    
df_results = pd.DataFrame(query_results)

print("Top 10 most relevant weather chunks:\n")
print("=" * 80)
    
for idx, row in df_results.iterrows():
    print(f"\n#{idx + 1} | similarity: {row['similarity']:.4f}")
    print(f"Location: {row['location']}")
    print(f"Type: {row['source_type']} | Headline: {row['headline']}")
    print(f"Effective: {row['effective_at']}")
    print(f"Text: {row['chunk_text'][:200]}...")
    print("-" * 80)

print("\n✅ Vector search test completed successfully!")
