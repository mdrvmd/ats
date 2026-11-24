import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Recruiting Agent", page_icon="🤖", layout="wide")
st.title("AI Recruiting Agent")
st.caption("Подбор кандидатов под вакансию: TF-IDF, Sentence-BERT и LLM с объяснениями")
with st.sidebar:
    st.header("📥 Почта")
    if st.button("Забрать резюме из почты"):
        with st.spinner("Проверяем почту..."):
            resp = requests.post(f"{API_URL}/ingest", timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Новых резюме: {len(data['new_files'])}. Всего из почты: {data['email_candidates']}")
        else:
            st.error(f"Ошибка {resp.status_code}: {resp.text}")

# --- список вакансий из API ---
try:
    vacancies = requests.get(f"{API_URL}/vacancies", timeout=10).json()
except requests.RequestException:
    st.error("API недоступен. Запустите сервер: uvicorn src.api:app --reload")
    st.stop()

# --- выбор вакансии ---
mode = st.radio("Источник вакансии", ["Выбрать из списка", "Ввести текст"], horizontal=True)
params = {}
if mode == "Выбрать из списка":
    options = {f'{v["id"]}. {v["title"]}': v["id"] for v in vacancies}
    choice = st.selectbox("Вакансия", list(options))
    params["job_id"] = options[choice]
else:
    params["text"] = st.text_area(
        "Текст вакансии", height=150,
        placeholder="Например: Data Analyst. SQL, Python, Power BI, опыт от 2 лет...",
    )

# --- выбор метода ---
strategies = {
    "Sentence-BERT (по смыслу)": "semantic",
    "TF-IDF (по ключевым словам)": "tfidf",
    "LLM с объяснениями (~40 секунд)": "llm",
}
label = st.selectbox("Метод подбора", list(strategies))
params["strategy"] = strategies[label]
params["top_k"] = st.slider("Сколько кандидатов показать", 1, 10, 5)

# --- поиск ---
if st.button("Найти кандидатов", type="primary"):
    if mode == "Ввести текст" and not params["text"].strip():
        st.warning("Введите текст вакансии")
        st.stop()

    with st.spinner("Ищем кандидатов..."):
        resp = requests.get(f"{API_URL}/recommendations", params=params, timeout=300)

    if resp.status_code != 200:
        st.error(f"Ошибка {resp.status_code}: {resp.text}")
    else:
        for i, c in enumerate(resp.json()["candidates"], start=1):
            with st.container(border=True):
                title = c["name"] if c.get("source") == "email" and c.get("name") else f"Кандидат #{c['candidate_id']}"
                st.subheader(f"{i}. {title} · {c['category']}")
                if c.get("email"):
                    st.caption(f"📧 {c['email']}")
                col1, col2 = st.columns(2)
                col1.metric("Сходство (поиск)", f"{c['score']:.2f}")
                if "llm_score" in c:
                    col2.metric("Оценка LLM", f"{c['llm_score']:.1f}")
                    st.info(c["explanation"])
                with st.expander("Начало резюме"):
                    st.write(c["preview"])