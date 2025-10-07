# Removes html tags, markdown, string literals, and whitespace from a job posting
import re
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # 2. Remove Markdown bold/italic markers (**text**, *text*, __text__)
    text = re.sub(r"\*{1,2}|_{1,2}", "", text)
    # 3. Replace literal '\n' and '\t' with space
    text = text.replace("\\n", " ").replace("\\t", " ")
    # 4. Replace any remaining whitespace (spaces, tabs, newlines) with single space
    text = re.sub(r"\s+", " ", text)
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    return text