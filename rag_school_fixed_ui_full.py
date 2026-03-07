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

DEFAULT_QUICK_QUESTIONS = [
    "입사신청기간은?",
    "합격자 발표일은?",
    "생활관비 납부기간은?",
    "제출서류는 무엇인가요?",
    "입사절차를 순서대로 알려줘",
    "입사등록할 때 준비물은?",
    "문의 전화번호는?"
]

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("OPENAI_API_KEY가 없습니다. Streamlit Secrets 또는 .env를 확인하세요.")

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="학교 문서 챗봇", page_icon="📚")
st.title("📚 학교 문서 챗봇")
st.caption("서버에 등록된 PDF를 자동으로 읽어 답변합니다.")

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
        expanded.extend([
            "인터넷 입사신청",
            "입사 신청 기간",
            "신청 일정",
            "모집일정"
        ])

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
        expanded.extend([
            "증빙서류",
            "제출서류",
            "준비물",
            "입사원서",
            "결핵검진결과표"
        ])

    if "일정" in q:
        expanded.extend([
            "모집일정",
            "신청기간",
            "합격자 발표",
            "납부기간",
            "입사개시일"
        ])

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


def ask_gpt(question, selected_results, all_pages, selected_user_type):
    content = build_multimodal_input(question, selected_results, all_pages, selected_user_type)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 학교 문서 안내 도우미다. "
                    "답변은 정확하고 간결하게 작성하라. "
                    "사용자가 선택한 유형에 맞는 내용만 우선 안내하라. "
                    "질문이 전화번호, 연락처, 문의처 관련이면 문서에 있는 번호를 빠짐없이 정리해서 답하라."
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
    try:
        creds_info = dict(st.secrets["gcp_service_account"])
    except Exception:
        return None

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
    gc = get_gspread_client()
    if gc is None:
        return None

    try:
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]
    except Exception:
        return None

    sh = gc.open(sheet_name)
    return sh.sheet1


def append_google_sheet_log(question, selected_user_type, sources_text="", answer_preview=""):
    try:
        ws = get_log_worksheet()
        if ws is None:
            return False

        ws.append_row([
            datetime.now().isoformat(timespec="seconds"),
            question,
            selected_user_type,
            sources_text,
            answer_preview[:200]
        ])
        return True
    except Exception as e:
        print("Google Sheets 로그 저장 실패:", e)
        return False


def read_google_sheet_logs():
    try:
        ws = get_log_worksheet()
        if ws is None:
            return []
        rows = ws.get_all_records()
        return rows
    except Exception as e:
        print("Google Sheets 로그 읽기 실패:", e)
        return []


def append_question_log(question, selected_user_type, sources_text="", answer_preview=""):
    append_local_question_log(question, selected_user_type, sources_text, answer_preview)
    append_google_sheet_log(question, selected_user_type, sources_text, answer_preview)


def read_question_logs():
    sheet_logs = read_google_sheet_logs()
    if sheet_logs:
        return sheet_logs
    return read_local_question_logs()


def get_question_stats():
    logs = read_question_logs()
    counter = {}

    for row in logs:
        q = str(row.get("question", "")).strip()
        if q:
            counter[q] = counter.get(q, 0) + 1

    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


def get_top_questions(limit=6):
    stats = get_question_stats()
    return [q for q, _ in stats[:limit]]


def normalize_question_for_button(q: str) -> str:
    return re.sub(r"\s+", "", q.strip()).lower()


def build_quick_questions():
    top_logged_questions = get_top_questions(limit=6)

    if top_logged_questions:
        base_questions = top_logged_questions + DEFAULT_QUICK_QUESTIONS
    else:
        base_questions = DEFAULT_QUICK_QUESTIONS

    seen = set()
    deduped = []

    for q in base_questions:
        nq = normalize_question_for_button(q)
        if nq not in seen:
            seen.add(nq)
            deduped.append(q)

    return deduped[:9]


# -----------------------------
# 캐시 저장 / 불러오기
# -----------------------------
def load_cached_rag_data():
    if not os.path.exists(RAG_CACHE_FILE):
        return None

    try:
        with open(RAG_CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data
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

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("본인 유형 선택")

    selected_user_type = st.selectbox(
        "사용자 유형",
        FIXED_CATEGORIES
    )

    st.caption("처음 화면의 선택 항목은 고정되어 있습니다.")

    st.markdown("---")
    st.subheader("운영 도구")

    if st.button("PDF 다시 불러오기"):
        st.cache_resource.clear()
        if os.path.exists(RAG_CACHE_FILE):
            os.remove(RAG_CACHE_FILE)
        st.rerun()

    logs = read_question_logs()
    st.caption(f"누적 질문 로그 수: {len(logs)}")
    st.caption(f"캐시 사용 여부: {'예' if cache_hit else '아니오'}")

    st.markdown("---")
    st.subheader("질문 통계")
    stats = get_question_stats()[:10]
    if stats:
        for q, count in stats:
            st.write(f"{q} ({count})")
    else:
        st.caption("아직 통계가 없습니다.")

st.info(f"현재 선택 유형: {selected_user_type}")
st.caption(f"등록된 PDF 페이지 수: {len(all_pages)} | 검색용 문서 수: {len(docs)}")

# -----------------------------
# FAQ / 빠른 질문
# -----------------------------
quick_questions = build_quick_questions()

st.subheader("빠른 질문")
st.caption("자주 묻는 질문과 기본 질문을 버튼으로 제공합니다.")

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
        if msg.get("user_type"):
            st.caption(f"선택 유형: {msg['user_type']}")
        if msg.get("sources_text"):
            st.caption(f"참고 자료: {msg['sources_text']}")

# -----------------------------
# 질문 입력
# -----------------------------
typed_prompt = st.chat_input("질문을 입력하세요")

prompt = ""
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = ""
elif typed_prompt:
    prompt = typed_prompt

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "user_type": selected_user_type
    })

    filtered_docs, filtered_embeddings = filter_docs_and_embeddings(
        docs,
        doc_embeddings,
        selected_user_type
    )

    if not filtered_docs:
        answer = f"선택한 유형({selected_user_type})에 해당하는 문서를 찾지 못했습니다."

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "user_type": selected_user_type,
            "sources_text": ""
        })

        append_question_log(
            question=prompt,
            selected_user_type=selected_user_type,
            sources_text="",
            answer_preview=answer
        )
    else:
        search_question = prompt
        if any(k in prompt for k in ["전화", "번호", "연락처", "문의"]):
            search_question = prompt + " 전화번호 연락처 문의처 행정팀 사감실 02-"

        with st.spinner("관련 페이지 검색 중..."):
            selected_results = retrieve_top_results(
                question=search_question,
                docs=filtered_docs,
                doc_embeddings=filtered_embeddings,
                top_k=6
            )

        with st.spinner("답변 생성 중..."):
            answer, sources_text = ask_gpt(
                question=prompt,
                selected_results=selected_results,
                all_pages=all_pages,
                selected_user_type=selected_user_type
            )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"선택 유형: {selected_user_type}")
            st.caption(f"참고 자료: {sources_text}")

            with st.expander("참고한 파일/페이지 보기"):
                for item in selected_results:
                    page_info = find_page_info(
                        item["source_file"],
                        item["page_num"],
                        all_pages
                    )

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
                            st.text(page_info["text"][:2000])

                    st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "user_type": selected_user_type,
            "sources_text": sources_text
        })

        append_question_log(
            question=prompt,
            selected_user_type=selected_user_type,
            sources_text=sources_text,
            answer_preview=answer
        )
