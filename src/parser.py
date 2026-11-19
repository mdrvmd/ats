import pdfplumber
from docx import Document
import re
from pathlib import Path

EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE_RE = r"\+?\d[\d\s()-]{8,}\d"
SECTION_HEADERS = {
    "experience": ["опыт работы", "опыт", "experience", "work experience", "professional experience"],
    "education":  ["образование", "education"],
    "skills":     ["навыки", "ключевые навыки", "skills", "technical skills", "технические навыки"],
}

def extract_text(path):
    if path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    elif path.lower().endswith(".docx"):
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text
    else:
        raise ValueError(f"Неподдерживаемый формат: {path}")


def extract_contacts(text):
    emails = re.findall(EMAIL_RE, text)
    phones = re.findall(PHONE_RE, text)
    phones = [p for p in phones if len(re.sub(r"\D", "", p)) >= 10]
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
    }

SKILLS_PATH = Path(__file__).parent.parent / "data" / "skills.txt"

def load_skills(path=SKILLS_PATH):
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS = load_skills()
def extract_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return found


def detect_header(line):
    clean = line.strip().lower().rstrip(":")
    for section, names in SECTION_HEADERS.items():
        if clean in names:
            return section
    return None

def split_sections(text):
    sections = {"header": [], "experience": [], "education": [], "skills": []}
    current = "header"                     # всё до первого заголовка: имя, контакты
    for line in text.split("\n"):
        section = detect_header(line)
        if section:
            current = section              # нашли заголовок: переключаемся
        elif line.strip():
            sections[current].append(line.strip())
    return {name: "\n".join(lines) for name, lines in sections.items()}

def parse_resume(path):
    text = extract_text(path)
    contacts = extract_contacts(text)
    sections = split_sections(text)
    header_lines = sections["header"].split("\n")
    return {
        "name": header_lines[0] if header_lines else None,
        "email": contacts["email"],
        "phone": contacts["phone"],
        "skills": extract_skills(text),
        "experience": sections["experience"],
        "education": sections["education"],
        "raw_text": text,
    }

parse_resume("../data/resumes/Мадияров Мади cv (1).pdf")
