import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text_cleaning import clean_text

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"

# --- загружается один раз при импорте модуля ---
model = SentenceTransformer(EMBEDDING_MODEL)

df = pd.read_csv(DATA / "eval" / "Resume.csv")
df["clean_text"] = df["Resume_str"].apply(clean_text)

EMB_PATH = DATA / "eval" / "resume_embeddings.npy"
if EMB_PATH.exists():
    resume_embeddings = np.load(EMB_PATH)
else:
    resume_embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True)
    np.save(EMB_PATH, resume_embeddings)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def search(vacancy_text, k=5):
    vacancy_vec = model.encode([clean_text(vacancy_text)])
    scores = cosine_similarity(vacancy_vec, resume_embeddings)[0]
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


tfidf_vectorizer = TfidfVectorizer(token_pattern=r"\S+")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])


def search_tfidf(vacancy_text, k=5):
    vacancy_vec = tfidf_vectorizer.transform([clean_text(vacancy_text)])
    scores = cosine_similarity(vacancy_vec, tfidf_matrix)[0]
    result = df.copy()
    result["score"] = scores
    return result.sort_values("score", ascending=False).head(k)