import json

from fastapi import FastAPI, HTTPException, Query

from src.matching import DATA, rerank, search, search_tfidf

app = FastAPI(title="AI Recruiting Agent")

with open(DATA / "vacancies.json", encoding="utf-8") as f:
    VACANCIES = {v["id"]: v for v in json.load(f)}


def to_candidates(result, with_llm):
    candidates = []
    for _, row in result.iterrows():
        item = {
            "candidate_id": int(row["ID"]),
            "category": row["Category"],
            "score": round(float(row["score"]), 3),
            "preview": row["Resume_str"][:300].strip(),
        }
        if with_llm:
            item["llm_score"] = float(row["llm_score"])
            item["explanation"] = row["explanation"]
        candidates.append(item)
    return candidates


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vacancies")
def list_vacancies():
    return [{"id": v["id"], "title": v["title"]} for v in VACANCIES.values()]


@app.get("/recommendations")
def recommendations(
    job_id: int | None = None,
    text: str | None = None,
    strategy: str = Query("semantic", pattern="^(tfidf|semantic|llm)$"),
    top_k: int = Query(5, ge=1, le=20),
):
    if (job_id is None) == (text is None):
        raise HTTPException(status_code=400, detail="Укажите либо job_id, либо text")

    if job_id is not None:
        if job_id not in VACANCIES:
            raise HTTPException(status_code=404, detail=f"Вакансия {job_id} не найдена")
        vacancy_text = VACANCIES[job_id]["text"]
    else:
        vacancy_text = text

    if strategy == "tfidf":
        result, with_llm = search_tfidf(vacancy_text, k=top_k), False
    elif strategy == "semantic":
        result, with_llm = search(vacancy_text, k=top_k), False
    else:
        result, with_llm = rerank(vacancy_text, k=top_k), True

    return {"strategy": strategy, "candidates": to_candidates(result, with_llm)}