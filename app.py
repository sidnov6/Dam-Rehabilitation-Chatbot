"""
RAG chatbot for the Dam Rehabilitation Manual.
Run: streamlit run app.py
"""

import os, re, base64, shutil
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# On cloud (read-only source dir) copy chroma_db to /tmp so ChromaDB can write lock files
_SRC_CHROMA = os.path.join(os.path.dirname(__file__), "chroma_db")
_TMP_CHROMA = "/tmp/dam_chroma_db"
if not os.path.exists(_TMP_CHROMA) and os.path.exists(_SRC_CHROMA):
    shutil.copytree(_SRC_CHROMA, _TMP_CHROMA)
CHROMA_DIR = _TMP_CHROMA if os.path.exists(_TMP_CHROMA) else _SRC_CHROMA
COLLECTION   = "dam_rehab_manual"
TOP_K        = 6
MODEL        = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an expert assistant specializing in the rehabilitation of large dams. "
    "You answer questions strictly based on the provided context excerpts from the "
    "official manual 'Manual for Rehabilitation of Large Dams'.\n\n"
    "Guidelines:\n"
    "- Answer only from the provided context. If the answer is not in the context, say so clearly.\n"
    "- Cite the page number(s) when relevant (e.g., 'According to page 42...').\n"
    "- Use clear, technical but accessible language.\n"
    "- If the user's question is vague, ask a clarifying question.\n"
    "- Do not fabricate information beyond what the manual states."
)

# ── Suggested questions ───────────────────────────────────────────────────────
FEATURED = [
    "What are the most common causes of dam failure in India?",
    "What is piping in embankment dams and how is it repaired?",
    "What is ALARP in dam safety risk management?",
    "How can rehabilitation be done without emptying the reservoir?",
    "What are grout curtains and foundation drains?",
    "How does sonic tomography help in dam investigations?",
    "Compare rehabilitation methods for concrete dams vs embankment dams.",
    "Create a checklist for dam rehabilitation inspections.",
]

ALL_QUESTIONS = {
    ("🌊", "Beginner / General"): [
        "What are the most common causes of dam failure in India?",
        "Why do old dams require rehabilitation?",
        "What is the difference between maintenance and rehabilitation of dams?",
        "What are the major types of dams discussed in the manual?",
        "How does overtopping cause dam failure?",
    ],
    ("⚙️", "Technical / Engineering"): [
        "What field investigations are required before rehabilitating an embankment dam?",
        "What tests are performed on concrete and masonry dams?",
        "How is seepage controlled in concrete dams?",
        "What is piping in embankment dams and how is it repaired?",
        "What are grout curtains and foundation drains?",
        "What materials are recommended for concrete repair?",
        "What are the advantages of fiber reinforced concrete in rehabilitation?",
        "How does sonic tomography help in dam investigations?",
        "What are the rehabilitation methods for spillways?",
        "How are seismic risks evaluated for existing dams?",
    ],
    ("🛡️", "Risk & Safety"): [
        "What is ALARP in dam safety risk management?",
        "How is probabilistic risk analysis used in dam rehabilitation?",
        "What are the warning signs of structural distress in dams?",
        "Which dam failures in India were caused by overtopping?",
        "What role does instrumentation play in dam safety?",
    ],
    ("🔧", "Practical / Real-World"): [
        "How can rehabilitation be done without emptying the reservoir?",
        "What rehabilitation methods are used for seepage in old masonry dams?",
        "How are damaged spillway gates repaired?",
        "What precautions should be taken while using epoxy repair materials?",
        "What are the common rehabilitation methods for embankment erosion?",
    ],
    ("🔬", "Advanced / Deep Dive"): [
        "Compare rehabilitation methods for concrete dams vs embankment dams.",
        "Explain the full rehabilitation planning process from inspection to execution.",
        "What international standards are referenced in the manual?",
        "How are geophysical methods used to identify weak zones in dams?",
        "What are the biggest challenges in rehabilitating 100-year-old dams?",
    ],
    ("🏆", "Challenge Questions"): [
        "If a dam develops excessive seepage after 70 years, how would engineers diagnose and fix it?",
        "What would happen if spillway capacity is lower than revised PMF estimates?",
        "How would you rehabilitate a dam affected by earthquakes?",
        "Which rehabilitation techniques are considered most cost-effective?",
        "What lessons were learned from historical dam failures worldwide?",
    ],
    ("📋", "Structured Prompts"): [
        "Summarize Chapter 5 in simple language.",
        "Create a checklist for dam rehabilitation inspections.",
        "Explain seepage control methods with examples.",
        "List all laboratory tests mentioned for repair materials.",
        "Give me a table of rehabilitation techniques and their applications.",
    ],
}

# ── Image helper ──────────────────────────────────────────────────────────────
def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* ── App shell ── */
.stApp { background: #F7F7F5; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 860px !important;
    padding: 1.5rem 2rem 6rem !important;
    margin: 0 auto !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1E1E2E !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #CDD6F4 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {
    color: #89B4FA !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #BAC2DE !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] hr { border-color: rgba(205,214,244,0.12) !important; }
[data-testid="stSidebar"] label { color: #A6ADC8 !important; font-size: 0.78rem !important; }

/* Sidebar expanders */
[data-testid="stSidebar"] details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] summary {
    font-size: 0.78rem !important;
    color: #BAC2DE !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.75rem !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #A6ADC8 !important;
    border-radius: 8px !important;
    font-size: 0.74rem !important;
    text-align: left !important;
    padding: 0.45rem 0.7rem !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.2rem !important;
    line-height: 1.4 !important;
    transition: all 0.15s !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: rgba(137,180,250,0.1) !important;
    border-color: rgba(137,180,250,0.3) !important;
    color: #CDD6F4 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Hero image container ── */
.hero-wrap {
    position: relative;
    width: 100%;
    height: 280px;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.hero-img {
    width: 100%; height: 100%;
    object-fit: cover;
    object-position: center 60%;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(to right, rgba(15,23,42,0.82) 0%, rgba(15,23,42,0.5) 60%, rgba(15,23,42,0.2) 100%);
    display: flex; flex-direction: column; justify-content: center;
    padding: 2rem 2.5rem;
}
.hero-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #93C5FD; margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 1.85rem; font-weight: 700; color: #FFFFFF;
    line-height: 1.2; margin-bottom: 0.5rem;
    text-shadow: 0 1px 8px rgba(0,0,0,0.4);
}
.hero-sub {
    font-size: 0.88rem; color: #CBD5E1; max-width: 420px; line-height: 1.5;
}

/* ── Section title ── */
.section-title {
    font-size: 1rem; font-weight: 600; color: #111827;
    margin: 1.75rem 0 1rem;
}
.section-sub {
    font-size: 0.82rem; color: #6B7280;
    margin: -0.5rem 0 1rem;
}

/* ── Dam type cards (image grid) ── */
.card-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.75rem;
}
.dam-card {
    position: relative; border-radius: 12px; overflow: hidden;
    height: 130px; cursor: default;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    transition: transform 0.2s, box-shadow 0.2s;
}
.dam-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.18); }
.dam-card img { width: 100%; height: 100%; object-fit: cover; object-position: center; }
.dam-card-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(to top, rgba(15,23,42,0.85) 0%, rgba(15,23,42,0.2) 60%, transparent 100%);
    display: flex; flex-direction: column; justify-content: flex-end;
    padding: 0.6rem 0.75rem;
}
.dam-card-icon { font-size: 1rem; margin-bottom: 0.15rem; }
.dam-card-label { font-size: 0.7rem; font-weight: 600; color: #F1F5F9; letter-spacing: 0.3px; line-height: 1.2; }

/* ── Divider ── */
.hl { height: 1px; background: #E5E7EB; margin: 1.5rem 0; }

/* ── Main question buttons ── */
div[data-testid="stButton"] button {
    background: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    text-align: left !important;
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 3rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #3B82F6 !important;
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.12) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.65rem !important;
    border: 1px solid transparent !important;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #EFF6FF !important;
    border-color: #DBEAFE !important;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #FFFFFF !important;
    border-color: #F3F4F6 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1.5px solid #D1D5DB !important;
    background: #FFFFFF !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* ── Expander ── */
details { border: 1px solid #E5E7EB !important; border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
"""

# ── Core functions ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=COLLECTION, embedding_function=ef)


def retrieve_context(collection, query: str, top_k: int) -> tuple[str, list[int]]:
    results = collection.query(query_texts=[query], n_results=top_k)
    docs, metas = results["documents"][0], results["metadatas"][0]
    pages = sorted({m["page"] for m in metas})
    parts = [f"[Page {m['page']}]\n{d}" for d, m in zip(docs, metas)]
    return "\n\n---\n\n".join(parts), pages


def chat_with_groq(api_key, history, context, question):
    client = Groq(api_key=api_key)
    aug = (
        f"Use the following excerpts from the manual to answer the question.\n\n"
        f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Question: {question}"
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[:-1]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": aug})
    resp = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=2048, temperature=0.2)
    return resp.choices[0].message.content


def handle_prompt(prompt, collection, top_k, show_sources):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            ctx, pages = retrieve_context(collection, prompt, top_k)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            try:
                answer = chat_with_groq(GROQ_API_KEY, history, ctx, prompt)
            except RateLimitError as e:
                wait = re.search(r"try again in ([\d.]+)s", str(e))
                wm = f" Please wait ~{int(float(wait.group(1)))}s." if wait else ""
                st.error(f"**Rate limit reached.**{wm} Groq allows 6,000 free requests/day.")
                st.session_state.messages.pop()
                st.stop()
        st.markdown(answer)
        if pages:
            st.caption(f"📄 Pages referenced: {', '.join(str(p) for p in pages)}")
        if show_sources:
            with st.expander("View source excerpts"):
                st.markdown(ctx)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": ctx})


# ── HTML blocks ───────────────────────────────────────────────────────────────
def hero_html():
    img = b64("assets/hero_banner.jpg")
    return f"""
<div class="hero-wrap">
  <img class="hero-img" src="data:image/jpeg;base64,{img}" alt="Sardar Sarovar Dam"/>
  <div class="hero-overlay">
    <div class="hero-label">Official CWC / DRIP Manual · India</div>
    <div class="hero-title">Dam Rehabilitation<br>Manual Assistant</div>
    <div class="hero-sub">AI-powered Q&amp;A on inspection, repair, risk analysis &amp; rehabilitation of large dams — 290 pages, 887 knowledge chunks.</div>
  </div>
</div>"""


def dam_cards_html():
    cards = [
        ("assets/hoover_web.jpg",    "🏛️", "Concrete &amp; Masonry Dams"),
        ("assets/tehri_web.jpg",     "🏔️", "Embankment Dams"),
        ("assets/itaipu_web.jpg",    "💧", "Spillways &amp; Gates"),
        ("assets/bhakra_web.jpg",    "🛡️", "Risk &amp; Safety"),
    ]
    items = ""
    for path, icon, label in cards:
        img = b64(path)
        items += f"""
        <div class="dam-card">
          <img src="data:image/jpeg;base64,{img}" alt="{label}"/>
          <div class="dam-card-overlay">
            <div class="dam-card-icon">{icon}</div>
            <div class="dam-card-label">{label}</div>
          </div>
        </div>"""
    return f'<div class="card-grid">{items}</div>'


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dam Rehabilitation Manual — AI Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
try:
    collection = load_collection()
except Exception as e:
    st.error(f"Knowledge base not found. Run `python3 ingest.py` first.\n\n{e}")
    st.stop()

if not GROQ_API_KEY:
    st.error("Add `GROQ_API_KEY=...` to your `.env` file.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages = []
if "queued_prompt" not in st.session_state: st.session_state.queued_prompt = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Dam RAG Assistant")
    st.markdown("*Manual for Rehabilitation of Large Dams*")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved", 2, 12, TOP_K, label_visibility="collapsed")
    st.caption(f"Retrieval depth: **{top_k}** chunks")
    show_sources = st.toggle("Show source excerpts", value=False)

    st.markdown("---")
    st.markdown(f"📚 **{collection.count()} chunks** indexed  \n🤖 `{MODEL}`  \n📄 290 pages")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.queued_prompt = None
        st.rerun()

    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💡 More Questions")
        for (icon, cat), qs in ALL_QUESTIONS.items():
            with st.expander(f"{icon} {cat}"):
                for q in qs:
                    if st.button(q, key=f"sb_{hash(q)}", use_container_width=True):
                        st.session_state.queued_prompt = q
                        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(hero_html(), unsafe_allow_html=True)

if not st.session_state.messages:
    # Dam type image cards
    st.markdown(dam_cards_html(), unsafe_allow_html=True)

    # Featured questions
    st.markdown('<div class="section-title">What would you like to know?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Click any question or type your own below.</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, q in enumerate(FEATURED):
        with cols[i % 2]:
            if st.button(q, key=f"feat_{i}", use_container_width=True):
                st.session_state.queued_prompt = q
                st.rerun()

    st.markdown('<div class="hl"></div>', unsafe_allow_html=True)
else:
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                with st.expander("View source excerpts"):
                    st.markdown(msg["sources"])

# ── Handle suggestion click ───────────────────────────────────────────────────
if st.session_state.queued_prompt:
    p = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
    handle_prompt(p, collection, top_k, show_sources)
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if p := st.chat_input("Ask anything about dam rehabilitation, inspection, or repair..."):
    handle_prompt(p, collection, top_k, show_sources)
    st.rerun()
