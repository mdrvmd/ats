import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_cleaning import clean_text

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PARSED_DIR = DATA / "parsed"
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"

model = SentenceTransformer(EMBEDDING_MODEL)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"\S+")

SYSTEM_PROMPT = """You are an experienced recruiter. You compare a job vacancy with a candidate's resume and rate how well the candidate fits.

Scoring scale:
1.0 - perfect fit: same role, all key skills and relevant experience
0.7 - good fit: same role, minor gaps
0.4 - partial fit: related field, but important requirements are missing
0.1 - weak fit: different profession, only a few overlapping skills
0.0 - no relation at all

Be strict: do not give 0.7 or higher unless most key requirements are met.
Reply ONLY with JSON: {"score": <number from 0 to 1>, "explanation": "<2-3 sentences>"}
Write the explanation in the same language as the vacancy."""

# --- базовые кандидаты: датасет Resume.csv ---
base_df = pd.read_csv(DATA / "eval" / "Resume.csv")
base_df["clean_text"] = base_df["Resume_str"].apply(clean_text)
base_df["source"] = "dataset"

EMB_PATH = DATA / "eval" / "resume_embeddings.npy"
if EMB_PATH.exists():
    base_embeddings = np.load(EMB_PATH)
else:
    base_embeddings = model.encode(base_df["clean_text"].tolist(), show_progress_bar=True)
    np.save(EMB_PATH, base_embeddings)


# --- кандидаты из почты: data/parsed/*.json ---
def load_parsed_candidates():
    rows = []
    for i, path in enumerate(sorted(PARSED_DIR.glob("*.json"))):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "ID": 1_000_000 + i,
            "Category": "EMAIL",
            "Resume_str": parsed["raw_text"],
            "name": parsed.get("name"),
            "email": parsed.get("email"),
            "source": "email",
        })
    new_df = pd.DataFrame(rows, columns=["ID", "Category", "Resume_str", "name", "email", "source"])
    new_df["clean_text"] = new_df["Resume_str"].apply(clean_text)
    return new_df


def reload_candidates():
    global df, resume_embeddings, tfidf_matrix
    new_df = load_parsed_candidates()
    if len(new_df) > 0:
        new_embeddings = model.encode(new_df["clean_text"].tolist())
        df = pd.concat([base_df, new_df], ignore_index=True)
        resume_embeddings = np.vstack([base_embeddings, new_embeddings])
    else:
        df = base_df.copy()
        resume_embeddings = base_embeddings
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])
    return len(new_df)


reload_candidates()   # собрать кандидатов при старте


def search(vacancy_text, k=5):
    vacancy_vec = model.encode([clean_text(vacancy_text)])
    scores = cosine_similarity(vacancy_vec, resume_embeddings)[0]
    result = df.copy()
    result["score"] = scores
    return result.sort_values("score", ascending=False).head(k)


def search_tfidf(vacancy_text, k=5):
    vacancy_vec = tfidf_vectorizer.transform([clean_text(vacancy_text)])
    scores = cosine_similarity(vacancy_vec, tfidf_matrix)[0]
    result = df.copy()
    result["score"] = scores
    return result.sort_values("score", ascending=False).head(k)


def llm_score(vacancy_text, resume_text, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"VACANCY:\n{vacancy_text}\n\nRESUME:\n{resume_text[:4000]}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except RateLimitError:
            time.sleep(20)
    raise RuntimeError("LLM недоступна: лимит исчерпан")


def rerank(vacancy_text, k=5, shortlist=10):
    candidates = search(vacancy_text, k=shortlist)
    llm_scores, explanations = [], []
    for resume in candidates["Resume_str"]:
        result = llm_score(vacancy_text, resume)
        llm_scores.append(result["score"])
        explanations.append(result["explanation"])
    candidates["llm_score"] = llm_scores
    candidates["explanation"] = explanations
    return candidates.sort_values(["llm_score", "score"], ascending=False).head(k)