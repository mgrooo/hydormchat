import os
import re
import json
import pickle
import hashlib
import base64
import tempfile
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="한양대(서울) 학생생활관 챗봇(BETA)",
    page_icon="🏫",
    layout="wide"
)

# -----------------------------
# 기본 설정
# -----------------------------
PDF_FOLDER = "pdf"
LOG_FOLDER = "logs"
CACHE_FOLDER = "cache"

QUESTION_LOG_FILE = os.path.join(LOG_FOLDER, "question_log.jsonl")
RAG_CACHE_FILE = os.path.join(CACHE_FOLDER, "rag_cache.pkl")

FIXED_CATEGORIES = [
    "전체",
    "학부 신입생(수시/재외국민)",
    "학부 신입생(정시)",
    "학부 재학생",
    "대학원",
    "외국인 재학생"
]

CATEGORY_LABELS = {
    "전체": {"ko": "전체", "en": "All"},
    "학부 신입생(수시/재외국민)": {
        "ko": "학부 신입생(수시/재외국민)",
        "en": "Undergraduate Freshmen (Early / Overseas Koreans)"
    },
    "학부 신입생(정시)": {
        "ko": "학부 신입생(정시)",
        "en": "Undergraduate Freshmen (Regular Admission)"
    },
    "학부 재학생": {
        "ko": "학부 재학생",
        "en": "Current Undergraduate Students"
    },
    "대학원": {
        "ko": "대학원",
        "en": "Graduate Students"
    },
    "외국인 재학생": {
        "ko": "외국인 재학생",
        "en": "International Students"
    }
}

DEFAULT_QUICK_QUESTIONS = [
    "입사신청기간은?",
    "합격자 발표일은?",
    "생활관비 납부기간은?",
    "제출서류는 무엇인가요?",
    "입사절차를 순서대로 알려줘",
    "입사등록할 때 준비물은?",
    "문의 전화번호는?"
]

DEFAULT_QUICK_QUESTIONS_EN = [
    "When is the dorm application period?",
    "When will the admission results be announced?",
    "When is the dorm fee payment period?",
    "What documents are required?",
    "Could you explain the dorm application process step by step?",
    "What should I prepare before moving into the dorm?",
    "How can I contact the dormitory office?"
]

ENGLISH_SEARCH_MAP = {
    "When is the dorm application period?": "입사신청기간 모집일정 신청기간",
    "When will the admission results be announced?": "합격자 발표일 발표일정",
    "When is the dorm fee payment period?": "생활관비 납부기간 납부일정",
    "What documents are required?": "제출서류 증빙서류 준비물",
    "Could you explain the dorm application process step by step?": "입사절차 입사신청 절차 입사등록 절차",
    "What should I prepare before moving into the dorm?": "입사등록 준비물 제출서류 결핵검진결과표",
    "How can I contact the dormitory office?": "문의 전화번호 연락처 문의처 행정팀 사감실 02-"
}

ADMIN_PASSWORD = "1234"

# -----------------------------
# 환경 변수 / API
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("OPENAI_API_KEY가 없습니다. Streamlit Secrets 또는 .env를 확인하세요.")

try:
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", ADMIN_PASSWORD)
except Exception:
    pass

client = OpenAI(api_key=api_key)

# -----------------------------
# UI 스타일
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

/* 기본 버튼 */
.stButton > button {
    background-color: #0E4A84;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover {
    background-color: #1B6BB8;
    color: white;
}

/* select / alert */
div[data-baseweb="select"] > div {
    border-radius: 10px;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* expander / sidebar */
.streamlit-expanderHeader {
    font-weight: 700;
    color: #0E4A84;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0E4A84;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: #F7FAFC;
    padding: 10px;
    border-radius: 12px;
    border: 1px solid #D9E6F2;
    margin-bottom: 8px;
}

/* 상단 배너 */
.banner-card {
    background: linear-gradient(90deg, #0E4A84 0%, #1B6BB8 100%);
    padding: 22px 24px;
    border-radius: 18px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.banner-title {
    font-size: 2.1rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #FFFFFF;
    line-height: 1.25;
    margin-bottom: 8px;
}
.banner-subtitle {
    font-size: 1.08rem;
    font-weight: 700;
    color: #EAF3FF;
    line-height: 1.5;
}

/* 사용법 박스 */
.guide-box {
    background: #FFF6D8;
    border: 2px solid #D9A400;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 22px;
    font-size: 18px;
    line-height: 1.8;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
}
.guide-title {
    font-weight: 900;
    color: #8A5A00;
    margin-bottom: 10px;
    font-size: 1.28rem;
}
.required-badge {
    background: #D62828;
    color: white;
    font-size: 0.9rem;
    padding: 4px 10px;
    border-radius: 999px;
    margin-left: 8px;
    vertical-align: middle;
}
.guide-main {
    font-weight: 800;
    color: #1F3A5F;
    margin-bottom: 8px;
}
.guide-note {
    background: #FFFDF4;
    border: 1.5px solid #E7C96B;
    border-radius: 12px;
    padding: 12px 14px;
    color: #6B4E00;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.guide-warning {
    background: #FFF3F3;
    border: 1.5px solid #E09A9A;
    border-radius: 12px;
    padding: 12px 14px;
    color: #8B2E2E;
    font-size: 0.98rem;
    font-weight: 700;
}

/* 사이드바 카드 */
.sidebar-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 16px 14px;
    text-align: center;
    margin-bottom: 18px;
}
.sidebar-ko {
    font-size: 1.2rem;
    font-weight: 900;
    color: #0E4A84;
    margin-top: 6px;
}
.sidebar-en {
    font-size: 0.96rem;
    color: #4A5568;
    margin-top: 2px;
}
.fake-disabled {
    display: inline-block;
    width: 100%;
    text-align: center;
    background: #D1D5DB;
    color: #6B7280;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-weight: 700;
    margin-top: 10px;
    cursor: not-allowed;
    user-select: none;
    pointer-events: none;
}
.small-note {
    font-size: 0.88rem;
    color: #6B7280;
    margin-top: 8px;
    line-height: 1.5;
}

/* 채팅 박스 */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 6px 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 상단 배너
# -----------------------------
logo_path = "hanyang_logo.png"

col1, col2 = st.columns([1.2, 8])

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown(
            "<div style='font-size:58px; text-align:center;'>🏫</div>",
            unsafe_allow_html=True
        )

with col2:
    st.markdown("""
    <div class="banner-card">
        <div class="banner-title">한양대(서울) 학생생활관 챗봇(BETA)</div>
        <div class="banner-subtitle">
            학생생활관 모집요강 및 안내문서를 기반으로 답변합니다.<br>
            Answers are based on the dormitory recruitment guidelines and official documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 상단 사용법 안내
# -----------------------------

st.markdown("""
<div class="guide-box">
<div class="guide-title">
📌 사용법 / How to Use
<span class="required-badge">필수 / Required</span>
</div>

<div class="guide-main">
사용법 : (1) 사용언어 선택(한국어, 영어) (2) 사용자유형 선택 (3) 질문 입력
</div>

<div class="guide-main">
How to use: (1) Select language (Korean, English) (2) Select user type (3) Enter your question
</div>

<div class="guide-note">
※ 위 순서를 지켜야 더 정확한 답변이 가능합니다.<br>
※ Following these steps is required for more accurate answers.
</div>

<div class="guide-warning">
⚠ 중요사항은 반드시 원본 PDF도 함께 대조·확인해 주세요.<br>
⚠ For important matters, please always compare and confirm with the original PDF as well.
</div>
</div>
""", unsafe_allow_html=True)
# -----------------------------
# 유틸 함수
# -----------------------------
def ensure_directories():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_pdf_inventory():
    if not os.path.exists(PDF_FOLDER):
        return "no_pdf_folder"

    rows = []
    for filename in sorted(os.listdir(PDF_FOLDER)):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_FOLDER, filename)
        stat = os.stat(path)

        rows.append({
            "name": filename,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "md5": file_md5(path)
        })

    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def get_category_display_name(category_key: str, lang: str = "한국어") -> str:
    if category_key not in CATEGORY_LABELS:
        return category_key
    return CATEGORY_LABELS[category_key]["en"] if lang == "English" else CATEGORY_LABELS[category_key]["ko"]


def detect_auto_language(text: str) -> str:
    if not text:
        return "한국어"
    english_letters = len(re.findall(r"[A-Za-z]", text))
    korean_letters = len(re.findall(r"[가-힣]", text))
    if english_letters >= 5 and english_letters > korean_letters:
        return "English"
    return "한국어"


def save_uploaded_pdf(uploaded_file):
    ensure_directories()
    save_path = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path


def infer_fixed_category(filename: str, first_page_text: str = "") -> str:
    text = f"{filename} {first_page_text}".lower()

    if "외국인" in text or "international" in text or "non korean" in text:
        return "외국인 재학생"
    if "학부 재학생" in text:
        return "학부 재학생"
    if "대학원" in text or "일반대학원" in text or "전문대학원" in text:
        return "대학원"
    if "학부 신입생" in text and ("수시" in text or "재외국민" in text):
        return "학부 신입생(수시/재외국민)"
    if "학부 신입생" in text and "정시" in text:
        return "학부 신입생(정시)"
    if "정시" in text:
        return "학부 신입생(정시)"
    if "수시" in text or "재외국민" in text:
        return "학부 신입생(수시/재외국민)"
    return "기타"


def extract_pages_and_images_from_pdf_path(pdf_path: str, zoom: float = 2.0):
    pages = []
    filename = os.path.basename(pdf_path)

    file_stem = safe_filename(filename)
    temp_dir = tempfile.mkdtemp(prefix=f"{file_stem}_")

    with fitz.open(pdf_path) as doc:
        first_page_text = ""

        for i, page in enumerate(doc):
            text = normalize_text(page.get_text("text", sort=True))

            if i == 0:
                first_page_text = text

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            pix.save(image_path)

            pages.append({
                "source_file": filename,
                "page_num": i + 1,
                "text": text,
                "image_path": image_path,
            })

    user_type = infer_fixed_category(filename, first_page_text)

    for p in pages:
        p["user_type"] = user_type

    return pages


def split_page_into_chunks(page_text: str, chunk_size: int = 700, overlap: int = 120):
    chunks = []
    text = page_text.strip()
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += (chunk_size - overlap)

    return chunks


def build_documents(all_pages):
    docs = []

    for page in all_pages:
        docs.append({
            "source_file": page["source_file"],
            "page_num": page["page_num"],
            "text": page["text"] if page["text"] else "(텍스트 없음)",
            "doc_type": "page",
            "user_type": page["user_type"]
        })

        if page["text"]:
            for chunk_text in split_page_into_chunks(page["text"]):
                docs.append({
                    "source_file": page["source_file"],
                    "page_num": page["page_num"],
                    "text": chunk_text,
                    "doc_type": "chunk",
                    "user_type": page["user_type"]
                })

    return docs


def expand_query(query: str):
    expanded = [query]
    q = query.strip()

    if "입사신청기간" in q or ("신청" in q and "기간" in q):
        expanded.extend(["인터넷 입사신청", "입사 신청 기간", "신청 일정", "모집일정"])

    if "입사절차" in q or ("입사" in q and "절차" in q):
        expanded.extend([
            "입사신청 절차",
            "입사등록 절차",
            "입사등록(입사) 절차",
            "입사신청",
            "입사등록",
            "입사 방법",
            "입사 순서"
        ])

    if "서류" in q:
        expanded.extend(["증빙서류", "제출서류", "준비물", "입사원서", "결핵검진결과표"])

    if "일정" in q:
        expanded.extend(["모집일정", "신청기간", "합격자 발표", "납부기간", "입사개시일"])

    if "전화" in q or "번호" in q or "연락처" in q or "문의" in q:
        expanded.extend([
            "전화번호",
            "연락처",
            "문의처",
            "문의 전화번호",
            "담당부서 전화번호",
            "선발담당부서 전화번호",
            "행정팀 전화번호",
            "사감실 전화번호",
            "학생생활관 행정팀",
            "문의 및 연락처"
        ])

    return list(dict.fromkeys(expanded))


def get_embeddings_for_texts(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_boost_score(query_list, doc_text):
    text = doc_text or ""
    compact_text = text.replace(" ", "")
    score = 0.0

    for q in query_list:
        q2 = q.replace(" ", "")
        if q2 and q2 in compact_text:
            score += 0.25

        for token in re.findall(r"[가-힣A-Za-z0-9]+", q):
            token2 = token.replace(" ", "")
            if len(token2) >= 2 and token2 in compact_text:
                score += 0.05

    query_joined = " ".join(query_list)
    if any(k in query_joined for k in ["전화", "번호", "연락처", "문의"]):
        if "02-" in text:
            score += 0.40
        if "☎" in text:
            score += 0.35
        if re.search(r"02-\d{3,4}-\d{4}", text):
            score += 0.50

    return score


def filter_docs_and_embeddings(docs, doc_embeddings, selected_user_type):
    if selected_user_type == "전체":
        return docs, doc_embeddings

    filtered_docs = []
    filtered_embeddings = []

    for doc, emb in zip(docs, doc_embeddings):
        if doc.get("user_type") == selected_user_type:
            filtered_docs.append(doc)
            filtered_embeddings.append(emb)

    return filtered_docs, filtered_embeddings


def retrieve_top_results(question, docs, doc_embeddings, top_k=6):
    queries = expand_query(question)
    query_embeddings = get_embeddings_for_texts(queries)

    scored = []

    for doc, emb in zip(docs, doc_embeddings):
        sim_scores = [cosine_similarity(qe, emb) for qe in query_embeddings]
        best_sim = max(sim_scores)

        boost = keyword_boost_score(queries, doc["text"])
        page_bonus = 0.03 if doc["doc_type"] == "page" else 0.0
        final_score = best_sim + boost + page_bonus

        scored.append({
            "source_file": doc["source_file"],
            "page_num": doc["page_num"],
            "text": doc["text"],
            "doc_type": doc["doc_type"],
            "score": final_score,
            "user_type": doc["user_type"]
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    page_best = {}
    for item in scored:
        key = (item["source_file"], item["page_num"])
        if key not in page_best or item["score"] > page_best[key]["score"]:
            page_best[key] = item

    return sorted(page_best.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def find_page_info(source_file, page_num, all_pages):
    for page in all_pages:
        if page["source_file"] == source_file and page["page_num"] == page_num:
            return page
    return None


def format_sources(selected_results):
    grouped = {}
    for item in selected_results:
        grouped.setdefault(item["source_file"], set()).add(item["page_num"])

    parts = []
    for filename, pages in grouped.items():
        page_text = ", ".join(f"p.{p}" for p in sorted(pages))
        parts.append(f"{filename} ({page_text})")

    return " / ".join(parts)


def build_multimodal_input(question, selected_results, all_pages, selected_user_type):
    content = []

    intro_text = (
        f"다음은 '{selected_user_type}' 유형에 해당하는 PDF들에서 검색된 관련 페이지들이다. "
        "페이지 이미지와 추출 텍스트를 함께 보고 질문에 답하라. "
        "반드시 제공된 자료 안에서만 답하고, 없으면 '문서에서 확인되지 않습니다'라고 답하라. "
        "답변은 간결하고 정확하게 작성하라."
    )

    content.append({"type": "text", "text": intro_text})
    content.append({"type": "text", "text": f"질문: {question}"})

    for item in selected_results:
        page_info = find_page_info(item["source_file"], item["page_num"], all_pages)
        if not page_info:
            continue

        page_text = page_info["text"] if page_info["text"] else "(추출된 텍스트 없음)"
        data_url = image_file_to_data_url(page_info["image_path"])

        content.append({
            "type": "text",
            "text": (
                f"[유형: {page_info['user_type']} | 파일: {page_info['source_file']} | 페이지: {page_info['page_num']}]\n"
                f"추출 텍스트:\n{page_text}"
            )
        })

        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })

    return content


def ask_gpt(question, selected_results, all_pages, selected_user_type, answer_language="한국어"):
    content = build_multimodal_input(question, selected_results, all_pages, selected_user_type)

    if answer_language == "English":
        language_instruction = (
            "Answer in clear, natural, and helpful English. "
            "Focus especially on international students. "
            "If the source documents are in Korean, accurately translate the relevant details into English. "
            "Keep the answer concise and easy to understand."
        )
    else:
        language_instruction = "답변은 한국어로 정확하고 간결하게 작성하라."

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 학교 문서 안내 도우미다. "
                    "정확하고 간결하게 답하라. "
                    "사용자가 선택한 유형에 맞는 내용만 우선 안내하라. "
                    "질문이 전화번호, 연락처, 문의처 관련이면 문서에 있는 번호를 빠짐없이 정리해서 답하라. "
                    + language_instruction
                )
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    answer = response.choices[0].message.content.strip()
    sources_text = format_sources(selected_results)
    return answer, sources_text


def load_all_pdfs_from_folder():
    all_pages = []

    if not os.path.exists(PDF_FOLDER):
        return all_pages

    for filename in sorted(os.listdir(PDF_FOLDER)):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_FOLDER, filename)
        pages = extract_pages_and_images_from_pdf_path(path)
        all_pages.extend(pages)

    return all_pages


def get_ui_text(lang="한국어"):
    if lang == "English":
        return {
            "section_user_type": "Select User Type",
            "label_user_type": "User Type",
            "help_user_type": "Select your user type before asking so the chatbot can prioritize the relevant dormitory guide.",
            "section_tools": "Tools",
            "section_popular": "Popular Questions",
            "label_language": "Response Language",
            "caption_current": "Current selection",
            "metric_total": "Total Questions",
            "metric_cache": "Cache",
            "cache_yes": "Yes",
            "cache_no": "No",
            "no_stats": "No data yet.",
            "reload_button": "🔄 Reload PDFs",
            "admin_section": "Admin",
            "admin_password": "Admin Password",
            "admin_success": "Admin authenticated",
            "admin_error": "Invalid password",
            "upload_pdf": "Upload PDF",
            "upload_success": "Upload completed",
            "admin_panel": "### 🔧 Admin Panel",
            "loaded_pages": "Loaded PDF Pages",
            "searchable_docs": "Searchable Documents",
            "recent_logs": "Recent Logs",
            "faq_title": "⭐ Frequently Asked Questions",
            "faq_caption": "Click a question below to ask instantly. Common questions are updated automatically.",
            "guide": """
            <div style="
                background:#F7FAFC;
                border:1px solid #D9E6F2;
                border-radius:14px;
                padding:14px 16px;
                margin-bottom:18px;
            ">
                <b style="color:#0E4A84;">Guide</b><br>
                Ask about the dorm application period, result announcement, dorm fee payment,
                required documents, or contact information, and the chatbot will answer based on
                the uploaded dormitory guide documents.
            </div>
            """,
            "status_info_prefix": "Selected user type",
            "status_lang_prefix": "Response language",
            "status_caption": "Registered PDF pages",
            "status_caption2": "Searchable documents",
            "chat_input": "Enter your question",
            "no_docs": "No documents were found for the selected category",
            "search_spinner": "Searching relevant pages...",
            "answer_spinner": "Generating answer...",
            "expander": "View referenced files/pages",
            "extract_text": "Extracted Text",
            "user_type": "User type",
            "response_language": "Response language",
            "sources": "Sources",
            "top_questions_label": "Top Questions"
        }

    return {
        "section_user_type": "사용자 유형 선택",
        "label_user_type": "사용자 유형",
        "help_user_type": "질문 전에 본인 유형을 선택하면 해당 모집요강 중심으로 답변합니다.",
        "section_tools": "운영 도구",
        "section_popular": "자주 나온 질문",
        "label_language": "답변 언어",
        "caption_current": "현재 선택",
        "metric_total": "누적 질문 수",
        "metric_cache": "캐시 사용",
        "cache_yes": "예",
        "cache_no": "아니오",
        "no_stats": "아직 통계가 없습니다.",
        "reload_button": "🔄 PDF 다시 불러오기",
        "admin_section": "관리자 / Admin",
        "admin_password": "관리자 비밀번호 / Admin Password",
        "admin_success": "관리자 인증 완료 / Admin authenticated",
        "admin_error": "비밀번호가 올바르지 않습니다 / Invalid password",
        "upload_pdf": "PDF 업로드 / Upload PDF",
        "upload_success": "업로드 완료",
        "admin_panel": "### 🔧 관리자 패널",
        "loaded_pages": "불러온 PDF 페이지 수",
        "searchable_docs": "검색 문서 수",
        "recent_logs": "최근 로그 보기",
        "faq_title": "⭐ 자주 묻는 질문",
        "faq_caption": "버튼을 누르면 바로 질문할 수 있습니다. 최근 많이 들어온 질문이 자동 반영됩니다.",
        "guide": """
        <div style="
            background:#F7FAFC;
            border:1px solid #D9E6F2;
            border-radius:14px;
            padding:14px 16px;
            margin-bottom:18px;
        ">
            <b style="color:#0E4A84;">안내</b><br>
            입사신청기간, 합격자 발표, 생활관비 납부, 제출서류, 문의처 등을 질문하면
            등록된 모집요강 문서를 바탕으로 안내합니다.
        </div>
        """,
        "status_info_prefix": "현재 선택한 사용자 유형",
        "status_lang_prefix": "답변 언어",
        "status_caption": "등록된 PDF 페이지 수",
        "status_caption2": "검색용 문서 수",
        "chat_input": "질문을 입력하세요",
        "no_docs": "선택한 유형에 해당하는 문서를 찾지 못했습니다",
        "search_spinner": "관련 페이지 검색 중...",
        "answer_spinner": "답변 생성 중...",
        "expander": "참고한 파일/페이지 보기",
        "extract_text": "추출 텍스트",
        "user_type": "선택 유형",
        "response_language": "답변 언어",
        "sources": "참고 자료",
        "top_questions_label": "자주 나온 질문 통계"
    }


def is_english_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))

# -----------------------------
# 로컬 로그 저장
# -----------------------------
def append_local_question_log(question, selected_user_type, sources_text="", answer_preview=""):
    ensure_directories()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "user_type": selected_user_type,
        "sources": sources_text,
        "answer_preview": answer_preview[:200]
    }

    with open(QUESTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_local_question_logs():
    if not os.path.exists(QUESTION_LOG_FILE):
        return []

    rows = []
    with open(QUESTION_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

# -----------------------------
# Google Sheets
# -----------------------------
def get_gspread_client():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("secrets.toml에 [gcp_service_account] 가 없습니다.")

    creds_info = dict(st.secrets["gcp_service_account"])

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_info(
        creds_info,
        scopes=scopes
    )

    return gspread.authorize(credentials)


def get_log_worksheet():
    if "GOOGLE_SHEET_ID" not in st.secrets:
        raise ValueError("secrets.toml에 GOOGLE_SHEET_ID 가 없습니다.")

    gc = get_gspread_client()
    sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    sh = gc.open_by_key(sheet_id)
    ws = sh.get_worksheet(0)

    if not ws.cell(1, 1).value:
        ws.update("A1:E1", [[
            "timestamp",
            "question",
            "user_type",
            "sources",
            "answer_preview"
        ]])

    return ws


@st.cache_data(ttl=60, show_spinner=False)
def read_google_sheet_logs_cached(sheet_id: str):
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(sheet_id)
        ws = sh.get_worksheet(0)
        return ws.get_all_records()
    except Exception:
        return []


def append_google_sheet_log(question, selected_user_type, sources_text="", answer_preview=""):
    try:
        ws = get_log_worksheet()
        ws.append_row([
            datetime.now().isoformat(timespec="seconds"),
            question,
            selected_user_type,
            sources_text,
            answer_preview[:200]
        ])
        return True
    except Exception as e:
        st.error(f"Google Sheets 로그 저장 실패: {type(e).__name__}: {e}")
        return False


def read_google_sheet_logs():
    try:
        if "GOOGLE_SHEET_ID" not in st.secrets:
            return []
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        return read_google_sheet_logs_cached(sheet_id)
    except Exception:
        return []


def append_question_log(question, selected_user_type, sources_text="", answer_preview=""):
    append_local_question_log(question, selected_user_type, sources_text, answer_preview)
    append_google_sheet_log(question, selected_user_type, sources_text, answer_preview)
    read_google_sheet_logs_cached.clear()


def read_question_logs():
    sheet_logs = read_google_sheet_logs()
    if sheet_logs:
        return sheet_logs
    return read_local_question_logs()


def get_question_stats(logs=None):
    if logs is None:
        logs = read_question_logs()

    counter = {}
    for row in logs:
        q = str(row.get("question", "")).strip()
        if q:
            counter[q] = counter.get(q, 0) + 1

    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


def get_auto_faq_questions(limit=9, min_count=2, answer_language="한국어", logs=None):
    stats = get_question_stats(logs)

    filtered = []
    for q, count in stats:
        if count < min_count:
            continue
        if answer_language == "English":
            if is_english_text(q):
                filtered.append(q)
        else:
            if not is_english_text(q):
                filtered.append(q)

    return filtered[:limit]


def normalize_question_for_button(q: str) -> str:
    return re.sub(r"\s+", "", q.strip()).lower()


def build_quick_questions(answer_language="한국어", logs=None):
    top_logged_questions = get_auto_faq_questions(
        limit=9,
        min_count=2,
        answer_language=answer_language,
        logs=logs
    )

    default_questions = (
        DEFAULT_QUICK_QUESTIONS_EN if answer_language == "English"
        else DEFAULT_QUICK_QUESTIONS
    )

    if answer_language == "English":
        base_questions = default_questions + top_logged_questions
    else:
        base_questions = top_logged_questions + default_questions

    seen = set()
    deduped = []

    for q in base_questions:
        nq = normalize_question_for_button(q)
        if nq not in seen:
            seen.add(nq)
            deduped.append(q)

    return deduped[:6]

# -----------------------------
# 캐시 저장 / 불러오기
# -----------------------------
def load_cached_rag_data():
    if not os.path.exists(RAG_CACHE_FILE):
        return None

    try:
        with open(RAG_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cached_rag_data(pdf_inventory_signature, all_pages, docs, doc_embeddings):
    ensure_directories()
    cache_data = {
        "pdf_inventory_signature": pdf_inventory_signature,
        "all_pages": all_pages,
        "docs": docs,
        "doc_embeddings": doc_embeddings
    }

    with open(RAG_CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)


@st.cache_resource
def load_rag_data(pdf_inventory_signature: str):
    cached = load_cached_rag_data()

    if cached and cached.get("pdf_inventory_signature") == pdf_inventory_signature:
        return cached["all_pages"], cached["docs"], cached["doc_embeddings"], True

    all_pages = load_all_pdfs_from_folder()
    docs = build_documents(all_pages)

    doc_texts = [d["text"] if d["text"] else "(empty)" for d in docs]
    doc_embeddings = get_embeddings_for_texts(doc_texts)

    save_cached_rag_data(pdf_inventory_signature, all_pages, docs, doc_embeddings)
    return all_pages, docs, doc_embeddings, False

# -----------------------------
# 시작
# -----------------------------
ensure_directories()

pdf_inventory_signature = get_pdf_inventory()
all_pages, docs, doc_embeddings, cache_hit = load_rag_data(pdf_inventory_signature)

# -----------------------------
# 세션 상태
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "answer_language" not in st.session_state:
    st.session_state["answer_language"] = "한국어"

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

current_logs = read_question_logs()

# -----------------------------
# 본문 상단 사용자 설정 (모바일 우선)
# -----------------------------
current_ui_lang = st.session_state.get("answer_language", "한국어")
ui = get_ui_text(current_ui_lang)

st.markdown("""
<div style="
background:#F7FAFC;
border:1px solid #D9E6F2;
border-radius:16px;
padding:16px 18px;
margin-bottom:18px;
">
<div style="
font-size:1.08rem;
font-weight:800;
color:#0E4A84;
margin-bottom:8px;
">
사용자 설정 / User Settings
</div>

<div style="
font-size:0.95rem;
color:#4B5563;
line-height:1.6;
">
먼저 아래 항목을 선택해 주세요.<br>
Please select the options below first.
</div>
</div>
""", unsafe_allow_html=True)

selected_user_type = st.selectbox(
    "사용자 유형 선택 / Select User Type",
    FIXED_CATEGORIES,
    format_func=lambda x: get_category_display_name(x, current_ui_lang),
    key="selected_user_type_main"
)

answer_language = st.selectbox(
    "답변 언어 / Response Language",
    ["한국어", "English"],
    index=0 if current_ui_lang == "한국어" else 1,
    key="answer_language"
)

ui = get_ui_text(answer_language)

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:

    st.markdown("""
    <div class="sidebar-card">
    <div style="font-size:48px;">🏫</div>
    <div class="sidebar-ko">한양대(서울)</div>
    <div class="sidebar-en">Hanyang Univ. (Seoul)</div>

    <div class="sidebar-ko" style="margin-top:14px;">
    학생생활관 챗봇
    </div>

    <div class="sidebar-en">
    Dormitory Chatbot
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("현재 선택 / Current Selection")

    selected_user_type_display = get_category_display_name(
        selected_user_type,
        answer_language
    )

    st.caption(f"{selected_user_type_display} · {answer_language}")

    st.markdown(
        '<div class="fake-disabled">일반 사용자는 편집할 수 없습니다 / Editing disabled</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-note">위 설정을 정확히 선택하면 답변 정확도가 높아질 수 있습니다.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader(ui["section_tools"])

    is_admin = st.session_state.admin_authenticated

    if is_admin:
        if st.button(ui["reload_button"], use_container_width=True):

            st.cache_resource.clear()

            if os.path.exists(RAG_CACHE_FILE):
                os.remove(RAG_CACHE_FILE)

            st.rerun()

    st.metric(ui["metric_total"], len(current_logs))
    st.metric(ui["metric_cache"], ui["cache_yes"] if cache_hit else ui["cache_no"])

    st.markdown("---")
    st.subheader(ui["section_popular"])

    sidebar_stats = get_auto_faq_questions(
        limit=10,
        min_count=1,
        answer_language=answer_language,
        logs=current_logs
    )

    if sidebar_stats:

        sidebar_counter = {}
        all_stats = get_question_stats(current_logs)

        for q in sidebar_stats:
            for stat_q, count in all_stats:
                if stat_q == q:
                    sidebar_counter[q] = count
                    break

        for q, count in list(sidebar_counter.items())[:10]:
            st.write(f"• {q} ({count})")

    else:
        st.caption(ui["no_stats"])

    st.markdown("---")
    st.subheader(ui["admin_section"])

    admin_pw_input = st.text_input(
        ui["admin_password"],
        type="password"
    )

    if admin_pw_input:

        if admin_pw_input == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.success(ui["admin_success"])

        else:
            st.error(ui["admin_error"])

    if st.session_state.admin_authenticated:

        uploaded_file = st.file_uploader(
            ui["upload_pdf"],
            type=["pdf"]
        )

        if uploaded_file is not None:

            save_uploaded_pdf(uploaded_file)

            st.success(f"{ui['upload_success']}: {uploaded_file.name}")

            st.cache_resource.clear()

            if os.path.exists(RAG_CACHE_FILE):
                os.remove(RAG_CACHE_FILE)

            st.rerun()


# -----------------------------
# FAQ 버튼 (모바일 2열)
# -----------------------------
quick_questions = build_quick_questions(
    answer_language=answer_language,
    logs=current_logs
)

st.subheader(ui["faq_title"])
st.caption(ui["faq_caption"])

cols = st.columns(2)

for i, q in enumerate(quick_questions):

    with cols[i % 2]:

        if st.button(q, key=f"quick_{i}", use_container_width=True):

            st.session_state.pending_question = q
            st.rerun()

# -----------------------------
# 안내 박스
# -----------------------------
st.markdown(ui["guide"], unsafe_allow_html=True)

# -----------------------------
# 상태 표시
# -----------------------------
selected_user_type_display = get_category_display_name(selected_user_type, answer_language)
st.info(f"{ui['status_info_prefix']}: {selected_user_type_display} · {ui['status_lang_prefix']}: {answer_language}")
st.caption(f"{ui['status_caption']}: {len(all_pages)} · {ui['status_caption2']}: {len(docs)}")

# -----------------------------
# 관리자 패널
# -----------------------------
if st.session_state.admin_authenticated:
    st.markdown(ui["admin_panel"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(ui["loaded_pages"], len(all_pages))
    with col_b:
        st.metric(ui["searchable_docs"], len(docs))

    with st.expander(ui["recent_logs"]):
        for row in current_logs[-10:][::-1]:
            st.write(row)

    with st.expander(ui["top_questions_label"]):
        admin_stats = get_question_stats(current_logs)[:10]
        if admin_stats:
            for q, count in admin_stats:
                st.write(f"• {q} ({count})")
        else:
            st.caption("No logs yet." if answer_language == "English" else "아직 로그가 없습니다.")

# -----------------------------
# FAQ / 빠른 질문
# -----------------------------
quick_questions = build_quick_questions(answer_language=answer_language, logs=current_logs)

st.subheader(ui["faq_title"])
st.caption(ui["faq_caption"])

cols = st.columns(3)
for i, q in enumerate(quick_questions):
    with cols[i % 3]:
        if st.button(q, key=f"quick_{i}"):
            st.session_state.pending_question = q
            st.rerun()

# -----------------------------
# 이전 대화 표시
# -----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        msg_lang = msg.get("answer_language", "한국어")
        msg_ui = get_ui_text(msg_lang)

        if msg.get("user_type"):
            user_type_display = get_category_display_name(msg["user_type"], msg_lang)
            st.caption(f"{msg_ui['user_type']}: {user_type_display}")

        if msg.get("answer_language"):
            st.caption(f"{msg_ui['response_language']}: {msg['answer_language']}")

        if msg.get("sources_text"):
            st.caption(f"{msg_ui['sources']}: {msg['sources_text']}")

# -----------------------------
# 질문 입력
# -----------------------------
typed_prompt = st.chat_input(ui["chat_input"])

prompt = ""
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = ""
elif typed_prompt:
    prompt = typed_prompt
    auto_lang = detect_auto_language(typed_prompt)
    st.session_state["answer_language"] = auto_lang
    answer_language = auto_lang
    ui = get_ui_text(answer_language)

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "user_type": selected_user_type,
        "answer_language": answer_language
    })

    filtered_docs, filtered_embeddings = filter_docs_and_embeddings(
        docs,
        doc_embeddings,
        selected_user_type
    )

    selected_user_type_display = get_category_display_name(selected_user_type, answer_language)

    if not filtered_docs:
        answer = f"{ui['no_docs']} ({selected_user_type_display})."

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "user_type": selected_user_type,
            "sources_text": "",
            "answer_language": answer_language
        })

        append_question_log(
            question=prompt,
            selected_user_type=selected_user_type,
            sources_text="",
            answer_preview=answer
        )
    else:
        search_question = prompt

        if answer_language == "English":
            if prompt in ENGLISH_SEARCH_MAP:
                search_question = ENGLISH_SEARCH_MAP[prompt]
            elif any(k in prompt.lower() for k in ["phone", "contact", "inquiry", "number"]):
                search_question = prompt + " 문의 전화번호 연락처 문의처 행정팀 사감실 02-"
            else:
                lowered = prompt.lower()
                if "application" in lowered or "apply" in lowered:
                    search_question = prompt + " 입사신청 신청기간 모집일정"
                elif "result" in lowered or "announcement" in lowered:
                    search_question = prompt + " 합격자 발표 발표일정"
                elif "payment" in lowered or "fee" in lowered:
                    search_question = prompt + " 생활관비 납부기간 납부일정"
                elif "document" in lowered or "required" in lowered:
                    search_question = prompt + " 제출서류 증빙서류 준비물"
                elif "move in" in lowered or "check in" in lowered:
                    search_question = prompt + " 입사등록 준비물 결핵검진결과표"
        else:
            if any(k in prompt for k in ["전화", "번호", "연락처", "문의"]):
                search_question = prompt + " 전화번호 연락처 문의처 행정팀 사감실 02-"

        with st.spinner(ui["search_spinner"]):
            selected_results = retrieve_top_results(
                question=search_question,
                docs=filtered_docs,
                doc_embeddings=filtered_embeddings,
                top_k=6
            )

        with st.spinner(ui["answer_spinner"]):
            answer, sources_text = ask_gpt(
                question=prompt,
                selected_results=selected_results,
                all_pages=all_pages,
                selected_user_type=selected_user_type,
                answer_language=answer_language
            )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"{ui['user_type']}: {selected_user_type_display}")
            st.caption(f"{ui['response_language']}: {answer_language}")
            st.caption(f"{ui['sources']}: {sources_text}")

            with st.expander(ui["expander"]):
                for item in selected_results:
                    page_info = find_page_info(
                        item["source_file"],
                        item["page_num"],
                        all_pages
                    )

                    if answer_language == "English":
                        item_user_type_display = get_category_display_name(item["user_type"], "English")
                        st.markdown(
                            f"**Type: {item_user_type_display} | File: {item['source_file']} | "
                            f"Page: {item['page_num']} | Score: {item['score']:.4f} | Structure: {item['doc_type']}**"
                        )
                    else:
                        st.markdown(
                            f"**유형: {item['user_type']} | 파일: {item['source_file']} | "
                            f"페이지: {item['page_num']} | 점수: {item['score']:.4f} | 구조: {item['doc_type']}**"
                        )

                    if page_info:
                        st.image(
                            page_info["image_path"],
                            caption=f"{item['source_file']} - 페이지 {item['page_num']}",
                            use_container_width=True
                        )
                        if page_info["text"]:
                            st.text_area(
                                ui["extract_text"],
                                value=page_info["text"][:2000],
                                height=220,
                                key=f"text_{item['source_file']}_{item['page_num']}_{item['doc_type']}"
                            )

                    st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "user_type": selected_user_type,
            "sources_text": sources_text,
            "answer_language": answer_language
        })

        append_question_log(
            question=prompt,
            selected_user_type=selected_user_type,
            sources_text=sources_text,
            answer_preview=answer
        )





