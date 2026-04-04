import re
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoPeftModelForCausalLM = None


CLICKHOUSE_SCHEMA = """
Table: events (database: sentient_log)
Columns:
  - event_id     UUID
  - timestamp    DateTime64(3, 'UTC')
  - event_type   LowCardinality(String)
  - url          String
  - latency_ms   Float32
  - user_agent   String
  - metadata     JSON

Engine: MergeTree
ORDER BY (timestamp, event_type)
PARTITION BY toYYYYMM(timestamp)
""".strip()


SYSTEM_PROMPT = f"""You are a ClickHouse SQL expert embedded inside the SentientLog observability platform.

Given the following schema:
{CLICKHOUSE_SCHEMA}

Rules:
1. Output ONLY a single valid ClickHouse SQL query.
2. No markdown fences, no explanation, no extra text.
3. Default to last 24 hours when user does not provide a time range.
4. Add LIMIT 100 unless user asks otherwise.
5. Never generate destructive operations (DROP, ALTER, TRUNCATE, DELETE, INSERT, UPDATE, CREATE).
6. Always terminate with a semicolon.

Few-shot examples:
Example question: average latency last hour
Example SQL: SELECT avg(latency_ms) AS avg_latency_ms FROM sentient_log.events WHERE timestamp >= now() - INTERVAL 1 HOUR LIMIT 100;

Example question: count events in last day
Example SQL: SELECT count(*) AS event_count FROM sentient_log.events WHERE timestamp >= now() - INTERVAL 1 DAY LIMIT 100;

Example question: count page_view events last hour
Example SQL: SELECT count(*) AS page_view_count FROM sentient_log.events WHERE event_type = 'page_view' AND timestamp >= now() - INTERVAL 1 HOUR LIMIT 100;
"""


class GenerateSQLRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    max_new_tokens: int = Field(default=220, ge=16, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)


class GenerateSQLResponse(BaseModel):
    sql: str


app = FastAPI(title="GRUB-BOT SQL API", version="1.0.0")

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _model_path() -> str:
    return os.getenv("GRUBBOT_MODEL_PATH", "models/grubbot-sshleifer-tiny-gpt2-v1")


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

    cleaned = cleaned.strip()
    # Keep only first statement up to the first semicolon, removing any extra text.
    semi_index = cleaned.find(";")
    if semi_index >= 0:
        cleaned = cleaned[: semi_index + 1]

    # Drop obvious non-SQL leading text before SELECT/WITH.
    match = re.search(r"(?is)\b(SELECT|WITH)\b", cleaned)
    if match:
        cleaned = cleaned[match.start() :]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def _is_valid_select_sql(sql: str) -> bool:
    upper = sql.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False
    forbidden = ("DROP ", "ALTER ", "TRUNCATE ", "DELETE ", "INSERT ", "UPDATE ", "CREATE ")
    return not any(token in upper for token in forbidden)


def _render_prompt(question: str) -> str:
    return (
        f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
        f"USER:\n{question}\n\n"
        "ASSISTANT:\nSQL: "
    )


def _ensure_model_loaded() -> None:
    global _model
    global _tokenizer

    if _model is not None and _tokenizer is not None:
        return

    model_path = _model_path()
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if AutoPeftModelForCausalLM is not None:
        try:
            _model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        except Exception:
            _model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        _model = AutoModelForCausalLM.from_pretrained(model_path)
    _model.to(_device)
    _model.eval()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-sql", response_model=GenerateSQLResponse)
def generate_sql(payload: GenerateSQLRequest) -> GenerateSQLResponse:
    try:
        _ensure_model_loaded()
        assert _model is not None
        assert _tokenizer is not None

        prompt = _render_prompt(payload.question)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=payload.max_new_tokens,
                do_sample=payload.temperature > 0,
                temperature=payload.temperature,
                pad_token_id=_tokenizer.eos_token_id,
            )

        generated = _tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        sql = _extract_sql(generated)
        if not sql:
            raise ValueError("Model returned empty SQL")

        if not _is_valid_select_sql(sql):
            raise ValueError(f"Model returned non-SELECT SQL: {sql}")

        return GenerateSQLResponse(sql=sql)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {exc}") from exc