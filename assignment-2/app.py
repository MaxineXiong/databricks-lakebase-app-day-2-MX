"""
Weather Data Sync Application:
- Serves a Flask API for weather data ingestion
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Fetches weather alerts and forecasts from the NWS API via weather_client.py

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os

from flask import Flask, jsonify, render_template, request

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Lazy-loaded singleton: the sentence-transformers model is heavy (~100 MB),
# so we load it once on first search request, not at import time.
_embedding_model = None


def _get_embedding_model():
    """Load and cache the embedding model (lazy init on first call)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface"
        )
    return _embedding_model


def _call_llm(prompt: str, max_tokens: int = 400) -> str:
    """Call a Databricks Foundation Model for text generation (RAG summary).
    Uses the SDK's built-in HTTP client (w.api_client.do) which handles all
    auth types (PAT, OAuth, service principal) automatically — no manual
    token extraction needed.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    model = os.environ.get("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

    response = w.api_client.do(
        "POST",
        f"/serving-endpoints/{model}/invocations",
        body={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
    )
    return response["choices"][0]["message"]["content"]


def _perform_vector_search(query: str, top_k: int, source_type: str = None):
    """Run pgvector cosine similarity search.
    Returns list[dict] of results or (jsonify_response, status_code) on error.
    Optional source_type filters results to 'alert' or 'forecast' only.
    """
    model = _get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()
    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) AS cnt FROM {WEATHER_EMBEDDINGS_TABLE_NAME}"
            )
            row = cur.fetchone()
            if row["cnt"] == 0:
                return jsonify({
                    "error": "No weather embeddings found."
                    " Sync weather data and run the embedding notebook first."
                }), 404

            # Build query with optional source_type filter
            sql = (
                f"SELECT d.id, d.location, d.source_type, d.headline, "
                f"d.narrative_text, e.chunk_text, "
                f"1 - (e.embedding <=> %s::vector) AS similarity "
                f"FROM {WEATHER_EMBEDDINGS_TABLE_NAME} e "
                f"JOIN {WEATHER_TABLE_NAME} d ON d.id = e.document_id"
            )
            params = [embedding_str]

            if source_type and source_type != "all":
                sql += " WHERE d.source_type = %s"
                params.append(source_type)

            sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
            params.extend([embedding_str, top_k])

            cur.execute(sql, tuple(params))
            raw_results = cur.fetchall()

    return [
        {
            "location": r["location"],
            "source_type": r["source_type"],
            "headline": r["headline"],
            "chunk_text": r["chunk_text"],
            "similarity": round(float(r["similarity"]), 4),
        }
        for r in raw_results
    ]


def _generate_rag_summary(query: str, results: list[dict]) -> str:
    """Generate an LLM natural-language summary of the top search results (basic RAG)."""
    if not results:
        return "No weather information found for this query."

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[{i}] {r['location']} (similarity: {r['similarity']:.2f})\n"
            f"Headline: {r['headline']}\n"
            f"Text: {r['chunk_text'][:500]}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a weather information assistant. Based on the following "
        "weather alerts and forecasts retrieved from a semantic search, "
        "provide a concise natural-language summary that answers the user's question.\n\n"
        f"User question: \"{query}\"\n\n"
        f"Retrieved weather information:\n{context}\n\n"
        "Summary:"
    )

    try:
        return _call_llm(prompt, max_tokens=400)
    except Exception as e:
        logger.warning(f"LLM summary generation failed: {e}")
        return f"Summary unavailable (LLM error: {e}). See raw results below."


def ensure_weather_table():
    """
    Create the weather documents table in Lakebase if it doesn't exist yet.
    This table stores weather alerts and forecasts fetched from the NWS API.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id VARCHAR(255) PRIMARY KEY,
            location VARCHAR(255) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            headline TEXT NOT NULL,
            narrative_text TEXT,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Weather data sync UI for fetching alerts and forecasts from NWS."""
    return render_template("weather.html")


@app.route("/search")
def search_page():
    """Vector search page."""
    return render_template("search.html")


@app.route("/weather/source_types", methods=["GET"])
def list_source_types():
    """Return distinct source_type values from the weather_documents table."""
    try:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT source_type FROM {WEATHER_TABLE_NAME} "
                    f"ORDER BY source_type"
                )
                types = [row["source_type"] for row in cur.fetchall()]
        return jsonify({"source_types": types})
    except Exception as e:
        logger.exception("Failed to list source_types")
        return jsonify({"source_types": []}), 200


@app.route("/weather/sync", methods=["POST"])
def sync_weather_from_nws():
    """
    Fetch weather data for locations and upsert into weather_documents.
    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    """
    ensure_weather_table()
    client = WeatherClient()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    body = request.json
    locations = body.get("locations", [])
    limit = body.get("limit")

    if not locations or not isinstance(locations, list):
        return jsonify({"error": "locations must be a non-empty list"}), 400

    total = 0
    for location in locations:
        if not isinstance(location, str):
            continue
        parts = [p.strip() for p in location.split(",")]
        if len(parts) != 2:
            continue
        city, state = parts
        if not city or not state:
            continue

        try:
            documents = client.fetch_weather_data(
                city=city, state=state, limit=limit
            )
            total += _upsert_weather_batch(documents)
        except Exception as e:
            logger.exception(f"Error fetching weather for {location}")
            continue

    return jsonify({"synced": total, "locations": locations})


@app.route("/weather/search", methods=["GET", "POST"])
def search_weather():
    """
    Semantic search over weather embeddings using pgvector cosine similarity.
    POST body: {"query": "risk of flooding near rivers", "top_k": 5}
    GET params: ?query=risk+of+flooding&top_k=5  (also returns an LLM RAG summary)
    """
    # --- Parse inputs based on method ---
    if request.method == "GET":
        query = request.args.get("query", "")
        source_type = request.args.get("source_type", "all")
        try:
            top_k = int(request.args.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
    else:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        body = request.json or {}
        query = body.get("query", "")
        top_k = body.get("top_k", 5)
        source_type = body.get("source_type", "all")

    # --- Validate query ---
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query must be a non-empty string"}), 400

    # --- Clamp top_k to 1-20 ---
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(20, top_k))

    # --- Perform vector search ---
    results = _perform_vector_search(query.strip(), top_k, source_type=source_type)
    if isinstance(results, tuple):
        return results  # Error response

    # --- POST: return raw results ---
    if request.method == "POST":
        return jsonify(results)

    # --- GET: return results + RAG summary ---
    summary = _generate_rag_summary(query.strip(), results)
    return jsonify({
        "query": query.strip(),
        "top_k": top_k,
        "results": results,
        "summary": summary,
    })


def _upsert_weather_batch(documents: list[dict]) -> int:
    """Upsert weather documents into the weather_documents table.
    
    Each document contains normalized weather alert or forecast data
    with location info (combines city and state into location field).
    """
    import json as _json
    from datetime import datetime

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                # Parse effective_at timestamp if present
                effective_at = doc.get("effective_at")
                if effective_at and isinstance(effective_at, str):
                    try:
                        effective_at = datetime.fromisoformat(
                            effective_at.replace("Z", "+00:00")
                        )
                    except Exception:
                        effective_at = None
                
                # Parse synced_at timestamp
                synced_at = doc.get("synced_at")
                if synced_at and isinstance(synced_at, str):
                    try:
                        synced_at = datetime.fromisoformat(
                            synced_at.replace("Z", "+00:00")
                        )
                    except Exception:
                        synced_at = datetime.utcnow()
                else:
                    synced_at = datetime.utcnow()
                
                # Combine city and state into location field
                location = f"{doc.get('city')}, {doc.get('state')}"

                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, source_type, headline,
                        narrative_text, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    RETURNING (xmax = 0) AS inserted
                    """,
                    (
                        doc.get("id"),
                        location,
                        doc.get("source_type"),
                        doc.get("headline"),
                        doc.get("narrative_text"),
                        effective_at,
                        _json.dumps(doc.get("payload", {})),
                        synced_at,
                    ),
                )
                # Only count if it was a new insert (not an update)
                result = cur.fetchone()
                if result and result['inserted']:
                    count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")