# Semantic Weather Search with RAG Summary

A weather data pipeline that fetches alerts and forecasts from the National Weather Service (NWS) API, chunks the narrative text, computes vector embeddings, stores them in a Lakebase (Databricks-managed Postgres) database with pgvector, and serves a semantic search interface with AI-powered summaries via a Flask web app.

The application provides two main features:

* **Weather Data Sync** - Fetch weather alerts and forecasts from the NWS API for any US city and load them into Lakebase for later semantic search.
* **Weather Semantic Search** - Semantic search over stored weather documents using pgvector cosine similarity, with optional filtering by source type (alerts vs. forecasts) and an AI-generated natural-language summary of the retrieved results powered by a Databricks Foundation Model LLM.

## Data Source

We chose the **National Weather Service (NWS) API** (`https://api.weather.gov`) because:

* It is free and public with no authentication or API key required.
* It provides rich, unstructured narrative text for both active alerts (flood warnings, severe thunderstorm warnings, etc.) and regular forecasts (detailed daily/nightly forecasts with temperature, wind, and precipitation details).
* The narrative text is well-suited for semantic search since it contains natural-language descriptions of weather conditions that benefit from embedding-based retrieval rather than keyword matching.
* It covers all US states and territories via a simple grid-point system.

The `WeatherClient` (`weather_client.py`) fetches two types of documents per location:

* **Alerts** - active weather alerts for the state (e.g., Flood Warnings, Severe Thunderstorm Warnings), normalized with description and instruction text combined into `narrative_text`.
* **Forecasts** - regular forecast periods for the location's NWS grid point, using the `detailedForecast` field as `narrative_text`.

## Schema Decisions

### `weather_documents` table

| Column | Type | Description |
| --- | --- | --- |
| `id` | VARCHAR(255) PK | NWS alert ID or SHA-256 hash of location+period for forecasts |
| `location` | VARCHAR(255) | Combined "City, ST" label |
| `source_type` | VARCHAR(50) | `alert` or `forecast` |
| `headline` | TEXT | Alert headline or forecast period name + short forecast |
| `narrative_text` | TEXT | Full narrative text (description + instruction for alerts, detailedForecast for forecasts) |
| `effective_at` | TIMESTAMPTZ | When the alert/forecast is effective |
| `payload` | JSONB | Raw API response for full fidelity |
| `synced_at` | TIMESTAMPTZ | When the document was synced |
| `created_at` | TIMESTAMPTZ | Row creation timestamp (default `now()`) |

Indexes: B-tree on `location` and `source_type`.

### `weather_embeddings` table

| Column | Type | Description |
| --- | --- | --- |
| `id` | VARCHAR(255) PK | `{document_id}_{chunk_index}` |
| `document_id` | VARCHAR(255) FK | References `weather_documents(id)` with `ON DELETE CASCADE` |
| `chunk_index` | INT | Zero-based chunk position within the document |
| `chunk_text` | TEXT | The chunked text segment |
| `embedding` | VECTOR(384) | 384-dimensional embedding vector |
| `model_name` | VARCHAR(255) | Model used to compute the embedding |
| `created_at` | TIMESTAMPTZ | Row creation timestamp (default `now()`) |

Indexes:
* **HNSW** on `embedding` using `vector_cosine_ops` (`m=16`, `ef_construction=64`) for fast approximate nearest neighbor search.
* **B-tree** on `document_id` for join performance during search and anti-join checks for unembedded documents.

### Chunking Parameters

* **Chunk size**: 800 characters
* **Chunk overlap**: 100 characters

These values were chosen to keep chunks small enough for precise retrieval (a single chunk maps to one weather alert section or forecast period) while the overlap ensures context isn't lost at chunk boundaries.

### Embedding Model

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Dimensions**: 384
* **Rationale**: Compact, fast to load (~100 MB), and produces high-quality embeddings for short-to-medium text passages. The 384-dimension vector size keeps storage and search efficient in pgvector. The model dimension is configurable via the notebook widget and matched in a `match/case` block so swapping models automatically resizes the `VECTOR(N)` column.

## AI-Powered RAG Summary

The search endpoint (GET `/weather/search`) implements a basic Retrieval-Augmented Generation (RAG) pipeline:

1. The user's query is embedded using the same `all-MiniLM-L6-v2` model.
2. pgvector cosine similarity search retrieves the top-k most relevant weather text chunks.
3. The retrieved chunks are assembled into a prompt with the user's question.
4. A Databricks Foundation Model LLM (`databricks-meta-llama-3-3-70b-instruct` by default) generates a concise natural-language summary answering the query based on the retrieved context.

The LLM is called via the Databricks SDK's built-in HTTP client (`w.api_client.do`), which handles all authentication types (PAT, OAuth, service principal) automatically. If the LLM call fails for any reason, the search results are still returned with a fallback error message in the summary field.

The model can be overridden with the `LLM_MODEL` environment variable.

## Source Type Filtering

Weather documents are categorized as either `alert` or `forecast`. The search UI includes a dropdown that lets users filter retrieval to one source type or search across all documents. The filter is passed as a `source_type` query parameter to the search endpoint and applied as a `WHERE` clause in the SQL query.

Additionally, each result card in the UI displays a colored tag indicating whether the result is an Alert (orange) or Forecast (green), making it easy to distinguish document types at a glance.

## How to Run the Pipeline End-to-End

### Prerequisites

1. A Databricks workspace with Lakebase Postgres provisioned.
2. A Databricks secret scope named `database` with a secret key `lakebase-url` containing the Lakebase connection URL (see `setup_secrets.py`).
3. Serverless compute (CPU) for running the notebook.

### Step 1: Store the Lakebase Connection Secret (one-time)

```python
python setup_secrets.py
# Paste your Lakebase URL when prompted, e.g.:
# postgresql://role:password@host:5432/databricks_postgres?sslmode=require
```

### Step 2: Run the Ingestion Notebook

Open `ingest_weather_embeddings` and run all cells in order:

| Cell | Purpose |
| --- | --- |
| 2 | Install Python dependencies (`sentence-transformers`, `psycopg2-binary`, `databricks-sdk`, etc.) |
| 3 | Restart Python kernel to pick up new packages |
| 5 | Set configuration parameters (table names, embedding model, chunk size, fetch limit) |
| 6 | Parse Lakebase connection details from the secret URL |
| 7 | Test the psycopg2 connection to Lakebase |
| 8 | Initialize the `weather_documents` and `weather_embeddings` tables with pgvector extension and HNSW index |
| 10 | Fetch weather data for 50 US cities (from `locations.csv`) via the NWS API |
| 11 | Batch-insert weather documents into Lakebase (upsert with `ON CONFLICT DO NOTHING`) |
| 12 | Query back the loaded documents to verify ingestion |
| 14 | Chunk the `narrative_text` into overlapping 800-char segments |
| 15 | Compute 384-dim embeddings on each chunk using `all-MiniLM-L6-v2` |
| 17 | Batch-insert chunk embeddings into `weather_embeddings` (upsert with `ON CONFLICT DO NOTHING`) |
| 19 | Test vector search with a sample query to verify end-to-end retrieval |

### Step 3: Deploy and Use the Flask App

The Flask app (`app.py`) provides two web pages and four API endpoints:

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Weather Data Sync page (`weather.html`) - fetch weather data from NWS and load into Lakebase |
| `/search` | GET | Weather Information Search page (`search.html`) - semantic search over weather embeddings with AI summary and source type filtering |
| `/weather/sync` | POST | API: sync weather data for given locations. Body: `{"locations": ["Chicago, IL"], "limit": 50}` |
| `/weather/search` | GET | API: semantic search with AI RAG summary. Query params: `?query=risk+of+flooding&top_k=5&source_type=all`. Returns `{query, top_k, results, summary}` |
| `/weather/search` | POST | API: semantic search (raw results only). Body: `{"query": "risk of flooding near rivers", "top_k": 5, "source_type": "all"}` |
| `/weather/source_types` | GET | API: list distinct `source_type` values from the database for dropdown population |

To run locally:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8000/` to sync weather data, and `http://localhost:8000/search` to search.

### Search API Examples

GET with AI summary and source type filter:

```bash
curl "http://localhost:8000/weather/search?query=severe+thunderstorms+with+heavy+rain&top_k=5&source_type=alert"
```

Response:

```json
{
  "query": "severe thunderstorms with heavy rain",
  "top_k": 5,
  "results": [
    {
      "location": "Chicago, IL",
      "source_type": "alert",
      "headline": "Severe Thunderstorm Warning issued August 25 at 3:22PM CDT...",
      "chunk_text": "...A severe thunderstorm warning has been issued for...",
      "similarity": 0.7234
    }
  ],
  "summary": "Based on the retrieved weather alerts, severe thunderstorms with heavy rain are expected in the Chicago area..."
}
```

POST for raw results only:

```bash
curl -X POST http://localhost:8000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "severe thunderstorms with heavy rain", "top_k": 5, "source_type": "alert"}'
```

Response:

```json
[
  {
    "location": "Chicago, IL",
    "source_type": "alert",
    "headline": "Severe Thunderstorm Warning issued August 25 at 3:22PM CDT...",
    "chunk_text": "...A severe thunderstorm warning has been issued for...",
    "similarity": 0.7234
  }
]
```

## Project Structure

```
assignment-2/
  app.py                      # Flask app: sync + search + source_types endpoints, RAG summary, lazy embedding model loader
  lakebase.py                 # Lakebase connection helpers (get_connection, run_query, run_write, init_schema)
  weather_client.py           # NWS API client (geocode, alerts, forecasts, normalization)
  locations.csv               # 50 US cities with lat/lon coordinates
  setup_secrets.py            # One-time script to store Lakebase URL in Databricks secrets
  requirements.txt            # Python dependencies
  ingest_weather_embeddings/  # Databricks notebook: full sync -> embed -> search pipeline
  templates/
    weather.html               # Weather data sync UI
    search.html                # Weather search UI with AI summary, source type filter, and result tags
```

## Known Limitations and Future Improvements

* **No incremental embedding updates**: The notebook re-chunks and re-embeds all documents on each run. An incremental pipeline that only embeds new or updated documents (via the anti-join pattern on `document_id`) would be more efficient at scale.
* **NWS API rate limits**: The NWS API requests a User-Agent header and may throttle aggressive callers. Adding retry logic with exponential backoff and request rate limiting would improve robustness for large location lists.
* **Geocoding dependency**: The `WeatherClient` uses the free Nominatim geocoder for city-to-lat/lon conversion when coordinates aren't provided. Nominatim has strict rate limits. Using a cached geocoding table or a paid geocoding service would be more reliable.
* **No authentication on the Flask app**: The web app is open. Adding Databricks App authentication (X-Forwarded-Email header-based) would secure the endpoints.
* **Embedding model loaded per-process**: The Flask app lazily loads the sentence-transformers model on first search request. In a multi-worker deployment, each worker would load its own copy. Using a shared model server or caching the model in a volume would reduce memory usage.
* **LLM dependency for summaries**: The AI summary relies on a Databricks Foundation Model being available in the workspace. If the model is unavailable or the workspace lacks serving endpoint permissions, the summary falls back to an error message while search results are still returned.
* **No result deduplication**: The vector search returns individual chunks. Multiple chunks from the same document may appear in results. A post-processing step to deduplicate by `document_id` (keeping the highest-scoring chunk) would improve result diversity.
* **No scheduled refresh**: Weather data changes frequently. Scheduling the notebook to run daily (e.g., via a Databricks job) would keep the search index current.
