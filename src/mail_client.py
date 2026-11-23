import json
import os
from pathlib import Path
from datetime import date, timedelta
from imap_tools import AND, MailBox, MailMessageFlags
from dotenv import load_dotenv

from src.parser import parse_resume

ROOT = Path(__file__).parent.parent
RESUMES_DIR = ROOT / "data" / "resumes"
PARSED_DIR = ROOT / "data" / "parsed"
ALLOWED = (".pdf", ".docx")

load_dotenv(ROOT / ".env")

def looks_like_resume(parsed):
    has_sections = bool(parsed["experience"]) or bool(parsed["education"])
    return has_sections or len(parsed["skills"]) >= 3

def fetch_resumes():
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    with MailBox("imap.gmail.com").login(os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD")) as mailbox:
        criteria = AND(seen=False, date_gte=date.today() - timedelta(days=3))
        for msg in mailbox.fetch(criteria, mark_seen=False):
            has_resume = False
            for att in msg.attachments:
                filename = Path(att.filename).name
                if not filename.lower().endswith(ALLOWED):
                    continue

                path = RESUMES_DIR / filename
                path.write_bytes(att.payload)

                parsed = parse_resume(str(path))
                if not looks_like_resume(parsed):
                    path.unlink()          # не резюме: удаляем сохранённый файл
                    continue
                parsed["source_email"] = msg.from_
                json_path = PARSED_DIR / (path.stem + ".json")
                json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

                saved.append(filename)
                has_resume = True

            if has_resume:
                mailbox.flag(msg.uid, MailMessageFlags.SEEN, True)

    return saved


if __name__ == "__main__":
    files = fetch_resumes()
    print(f"Новых резюме: {len(files)}")
    for f in files:
        print(" -", f)