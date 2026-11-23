"""BeemaSathi Streamlit UI — Your Insurance Companion."""

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from config import INSURERS, KNOWLEDGE_SOURCES, LLM_LOCAL
from rag.retriever import PolicyRetriever
from rag.generator import BimaSaathiGenerator
from agents.claim_predictor import ClaimPredictor
from agents.orchestrator import BeemaSathiOrchestrator
from ui.debug_tools import format_bytes, get_ollama_runtime


DEBUG_MODE = os.getenv("BEEMASATHI_DEBUG", "").lower() in {"1", "true", "yes", "on"}

st.set_page_config(
    page_title="BeemaSathi",
    page_icon="🛡️",
    layout="wide",
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_data" not in st.session_state:
    st.session_state.debug_data = None

# Sidebar
with st.sidebar:
    st.title("🛡️ BeemaSathi")
    st.markdown("*Health and motor insurance desk*")
    st.divider()

    st.subheader("Select Knowledge Source")
    insurer = st.selectbox(
        "Insurer / Regulator",
        ["Any"] + [v["name"] for v in KNOWLEDGE_SOURCES.values()],
    )
    policy = st.text_input("Policy Name (optional)", placeholder="e.g., Comprehensive")

    st.divider()
    st.subheader("Tools")
    tool_mode = st.radio("Mode", ["Ask Question", "Claim Predictor", "Upload Policy PDF"])

    debug_visible = False
    if DEBUG_MODE:
        st.divider()
        st.subheader("Developer Debugging")
        debug_visible = st.toggle(
            "Show live diagnostics",
            value=True,
            help="Shows local model, memory allocation, retrieval, and timing details.",
        )

    st.divider()
    st.markdown("""
    I can help you:
    - Navigate health and motor plans
    - Compare waiting periods and exclusions
    - Check claim-rejection risk

    *Informational only. Not legal advice.*
    """)

# Main area
st.title("🛡️ BeemaSathi")
st.caption("Your desk colleague for Indian health and motor insurance.")

# Initialize backend
@st.cache_resource
def get_retriever():
    return PolicyRetriever()

@st.cache_resource
def get_generator():
    return BimaSaathiGenerator()

@st.cache_resource
def get_predictor():
    return ClaimPredictor()

@st.cache_resource
def get_orchestrator():
    return BeemaSathiOrchestrator(
        retriever=get_retriever(),
        generator=get_generator(),
        predictor=get_predictor(),
    )

retriever = get_retriever()
generator = get_generator()
predictor = get_predictor()
orchestrator = get_orchestrator()


@st.cache_data(ttl=60)
def load_catalog():
    return get_retriever().store.get_catalog()


indexed_catalog = load_catalog()

with st.sidebar:
    st.divider()
    st.subheader("Indexed Knowledge")
    if indexed_catalog:
        insurers = {item["insurer"] for item in indexed_catalog}
        summary = (
            f"Ready with health and motor wordings from {len(insurers - {'irdai'})} insurers"
        )
        if "irdai" in insurers:
            summary += " plus IRDAI regulations"
        st.caption(summary + ". Ask for a plan by name when you need it.")
        selected_key = next(
            (key for key, value in KNOWLEDGE_SOURCES.items() if value["name"] == insurer),
            None,
        )
        if selected_key and selected_key not in insurers:
            st.warning(
                f"No {insurer} document is indexed yet, so questions about it "
                "will return a knowledge gap instead of an answer."
            )
    else:
        st.caption("No policy documents indexed yet.")


def source_label(source):
    """Build a user-facing citation label with document limitations."""
    insurer_name = KNOWLEDGE_SOURCES.get(
        source.get("insurer"),
        {"name": source.get("insurer", "Unknown").replace("_", " ").title()},
    )["name"]
    document_type = source.get("document_type", "policy_wording")
    parts = [
        insurer_name,
        source.get("policy_name", "Unknown").replace("_", " ").title(),
        source.get("section", "Unknown section"),
        document_type.replace("_", " ").title(),
    ]
    if source.get("page") is not None:
        parts.append(f"Page {source['page']}")
    return " | ".join(parts)


def render_debug_panel(target, debug_data):
    """Render the latest runtime snapshot in the dedicated right-side panel."""
    if not (DEBUG_MODE and debug_visible) or target is None:
        return

    debug_data = debug_data or {
        "stage": "Idle — waiting for a request",
        "events": [],
        "runtime": get_ollama_runtime(LLM_LOCAL),
        "retrieval": [],
        "model_metrics": {},
    }
    runtime = debug_data.get("runtime", {})
    active_models = runtime.get("active_models", [])
    configured_model = runtime.get("configured_model", LLM_LOCAL)
    selected_model = next(
        (model for model in active_models if model["name"] == configured_model),
        active_models[0] if active_models else None,
    )

    with target.container(border=True):
        st.subheader("Runtime debug")
        st.caption("Visible only when `BEEMASATHI_DEBUG=1`")
        st.info(debug_data.get("stage", "Idle — waiting for a request"))

        service = runtime.get("service", "unavailable")
        st.markdown(
            f"**Ollama:** {'Connected' if service == 'connected' else 'Unavailable'}"
        )
        model_name = selected_model["name"] if selected_model else configured_model
        loaded_size = (
            format_bytes(selected_model.get("loaded_size_bytes"))
            if selected_model
            else "Not loaded"
        )
        vram = (
            format_bytes(selected_model.get("vram_bytes"))
            if selected_model
            else "Not loaded"
        )

        st.metric("Connected model", model_name)
        memory_col, gpu_col = st.columns(2)
        memory_col.metric(
            "Process RAM",
            format_bytes(runtime.get("ollama_process_rss_bytes")),
        )
        gpu_col.metric("GPU/VRAM", vram)
        st.caption(f"Loaded model size: {loaded_size}")

        system_memory = format_bytes(runtime.get("system_memory_bytes"))
        st.caption(
            f"System memory: {system_memory}. On Apple Silicon, CPU and GPU share "
            "unified memory, so RAM and VRAM figures can overlap."
        )

        if runtime.get("error"):
            st.warning(f"Ollama runtime metrics unavailable: {runtime['error']}")
        elif not active_models:
            st.info("Ollama is connected, but no model is currently loaded.")

        events = debug_data.get("events", [])
        st.markdown("**Live request log**")
        if events:
            log_lines = []
            for event in events:
                duration = event.get("Duration (ms)")
                duration_text = f" ({duration} ms)" if duration is not None else ""
                log_lines.append(
                    f"[{event.get('Status', 'Info').upper()}] "
                    f"{event.get('Stage', 'Request')}{duration_text}\n"
                    f"  {event.get('Details', '')}"
                )
            st.code("\n".join(log_lines), language=None)
        else:
            st.code("[IDLE] Waiting for user interaction", language=None)

        retrieval = debug_data.get("retrieval", [])
        if retrieval:
            st.markdown("**Retrieved chunks**")
            for chunk in retrieval:
                st.caption(
                    f"#{chunk['Rank']} {chunk['Section']} · "
                    f"{chunk['Type']} · page {chunk.get('Page', '—')} · "
                    f"distance {chunk['Distance']}"
                )

        model_metrics = debug_data.get("model_metrics", {})
        if model_metrics:
            st.markdown("**LLM metrics**")
            token_col, output_col = st.columns(2)
            token_col.metric(
                "Prompt tokens",
                model_metrics.get("prompt_eval_count") or "—",
            )
            output_col.metric(
                "Output tokens",
                model_metrics.get("eval_count") or "—",
            )
            st.caption(
                f"Backend: {model_metrics.get('backend', 'unknown')} · "
                f"Reason: {model_metrics.get('done_reason') or '—'}"
            )


# Keep developer output in a dedicated right-side column.
if DEBUG_MODE and debug_visible:
    main_area, debug_area = st.columns([3.2, 1.15], gap="large")
    with debug_area:
        debug_placeholder = st.empty()
    render_debug_panel(debug_placeholder, st.session_state.debug_data)
else:
    main_area = st.container()
    debug_placeholder = None


def publish_debug(debug_data):
    """Persist and repaint the latest right-side debug snapshot."""
    if debug_placeholder is None:
        return
    st.session_state.debug_data = debug_data
    render_debug_panel(debug_placeholder, debug_data)


with main_area:
    # Tool: Claim Predictor
    if tool_mode == "Claim Predictor":
        st.header("🔮 Claim Rejection Predictor")
        st.markdown("Fill in your details to check rejection risk BEFORE filing.")

        col1, col2 = st.columns(2)
        with col1:
            cp_insurer = st.selectbox(
                "Insurer",
                list(INSURERS.keys()),
                format_func=lambda x: INSURERS[x]["name"],
            )
            cp_policy = st.text_input("Policy Name", value="comprehensive")
            cp_hospital = st.text_input("Hospital Name")
            cp_network = st.selectbox(
                "Is it a network/cashless hospital?",
                [None, True, False],
                format_func=lambda x: (
                    "Select" if x is None else ("Yes" if x else "No")
                ),
            )
            cp_procedure = st.text_input(
                "Procedure/Treatment",
                placeholder="e.g., Angioplasty",
            )

        with col2:
            cp_room_rent = st.number_input(
                "Room Rent per Day (₹)",
                min_value=0,
                value=0,
                step=500,
            )
            cp_si = st.number_input(
                "Sum Insured (₹)",
                min_value=0,
                value=500000,
                step=100000,
            )
            cp_ped = st.checkbox("Have Pre-existing Disease (Diabetes/BP/etc)?")
            cp_months = st.number_input(
                "Months since policy started",
                min_value=0,
                value=12,
            )
            cp_intimation = st.number_input(
                "Hours since admission (for intimation)",
                min_value=0,
                value=12,
            )

        if st.button("Predict Rejection Risk", type="primary"):
            publish_debug(
                {
                    "stage": "Running claim rule engine",
                    "events": [
                        {
                            "Stage": "Claim request",
                            "Status": "Running",
                            "Duration (ms)": None,
                            "Details": "Evaluating deterministic insurance rules",
                        }
                    ],
                    "runtime": get_ollama_runtime(LLM_LOCAL),
                    "model_metrics": {},
                }
            )
            prediction_started = time.perf_counter()
            with st.spinner("Analyzing..."):
                result = predictor.predict(
                    insurer=cp_insurer,
                    policy_name=cp_policy,
                    hospital_name=cp_hospital,
                    is_network_hospital=cp_network,
                    procedure=cp_procedure,
                    room_rent_per_day=cp_room_rent if cp_room_rent > 0 else None,
                    sum_insured=cp_si if cp_si > 0 else None,
                    has_ped=cp_ped,
                    months_since_policy=cp_months,
                    intimation_hours=cp_intimation,
                )
            prediction_duration_ms = round(
                (time.perf_counter() - prediction_started) * 1000,
                2,
            )

            risk = result["rejection_risk_percent"]
            level = result["risk_level"]

            if risk < 30:
                st.success(f"✅ Rejection Risk: {risk}% — {level}")
            elif risk < 60:
                st.warning(f"⚠️ Rejection Risk: {risk}% — {level}")
            else:
                st.error(f"🚨 Rejection Risk: {risk}% — {level}")

            st.markdown(f"**Summary:** {result['summary']}")

            if result["failed_checks"]:
                st.subheader("Issues Found")
                for check in result["failed_checks"]:
                    with st.expander(
                        f"❌ {check['check'].replace('_', ' ').title()}"
                    ):
                        st.write(check["message"])
                        st.info(f"**Action:** {check['action']}")

            st.subheader("📋 Documents You Need")
            for doc in result["documents_needed"]:
                st.checkbox(doc, key=doc)

            publish_debug(
                {
                    "stage": "Claim analysis complete",
                    "events": [
                        {
                            "Stage": "Claim rule engine",
                            "Status": "Completed",
                            "Duration (ms)": prediction_duration_ms,
                            "Details": f"{len(result['failed_checks'])} failed checks",
                        }
                    ],
                    "runtime": get_ollama_runtime(LLM_LOCAL),
                    "model_metrics": {
                        "backend": "rule-engine",
                        "model": "No LLM used",
                    },
                }
            )

    # Tool: Upload Policy PDF
    elif tool_mode == "Upload Policy PDF":
        st.header("📄 Policy Upload & Analysis")
        st.info(
            "Coming soon: Upload your policy PDF and get instant 'Gotcha' highlights."
        )

        uploaded = st.file_uploader("Upload Policy PDF", type=["pdf"])
        if uploaded:
            st.success(f"Uploaded: {uploaded.name}")
            st.markdown("""
            **Next steps:**
            1. Save the PDF under `data/raw_pdfs/` or `data/raw_policies/{insurer}/`.
            2. Run `./appctl.sh index` so it is chunked and embedded.
            3. Refresh this page to see it under **Indexed Knowledge**.
            """)

    # Tool: Ask Question (default chat)
    else:
        st.header("💬 Ask BeemaSathi")
        selected_insurer_key = next(
            (
                key
                for key, value in KNOWLEDGE_SOURCES.items()
                if value["name"] == insurer
            ),
            None,
        )
        if selected_insurer_key == "irdai":
            st.info(
                "IRDAI regulations provide regulatory context. They do not establish "
                "coverage under a specific insurance policy."
            )

        WELCOME = (
            "Hi, I'm BeemaSathi — think of me as a colleague at an Indian insurance "
            "desk. I help you navigate **health** and **motor** plans, compare what "
            "is actually written in the wording, and spot claim-rejection risks.\n\n"
            "Tell me the insurer and plan, or ask something like:\n"
            "- Compare waiting periods on two health plans\n"
            "- Does this car policy cover own damage?\n"
            "- What usually gets a cashless claim rejected?\n\n"
            "I only use official documents loaded in BeemaSathi. If a plan is not "
            "there, I will say so instead of guessing.\n\n"
            "*This is informational only. For a binding answer, check the policy "
            "wording or a licensed advisor.*"
        )
        if not st.session_state.messages:
            st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for source in msg["sources"]:
                            st.caption(source_label(source))

        user_started = any(msg["role"] == "user" for msg in st.session_state.messages)
        if not user_started:
            suggestions = {
                "Compare two health plans": (
                    "How should I compare two health plans on waiting periods and exclusions?"
                ),
                "Motor own-damage vs liability": (
                    "What is the difference between own-damage and liability-only motor cover?"
                ),
                "Claim rejection risks": (
                    "What usually causes a health insurance claim to get rejected?"
                ),
            }
            picked = st.pills(
                "Try asking:",
                list(suggestions.keys()),
                label_visibility="collapsed",
            )
            if picked:
                st.session_state.messages.append(
                    {"role": "user", "content": suggestions[picked]}
                )
                st.session_state.pending_prompt = suggestions[picked]
                st.rerun()

        if prompt := st.session_state.pop("pending_prompt", None):
            debug_events = [
                {
                    "Stage": "Request received",
                    "Status": "Running",
                    "Duration (ms)": None,
                    "Details": "Preparing policy filters",
                }
            ]
            request_started = time.perf_counter()

            insurer_key = None
            for key, value in KNOWLEDGE_SOURCES.items():
                if value["name"] == insurer:
                    insurer_key = key
                    break

            publish_debug(
                {
                    "stage": "Searching policy documents",
                    "events": debug_events,
                    "runtime": get_ollama_runtime(generator.model),
                    "retrieval": [],
                    "model_metrics": {},
                }
            )

            with st.chat_message("assistant"):
                answer_started = time.perf_counter()
                with st.spinner("Planning, searching, and checking evidence..."):
                    result = orchestrator.answer(
                        prompt,
                        insurer=insurer_key,
                        policy_name=policy if policy else None,
                        catalog=indexed_catalog,
                    )
                answer_duration_ms = round(
                    (time.perf_counter() - answer_started) * 1000,
                    2,
                )
                plan = result.get("plan") or {}
                verification = result.get("verification") or {}
                debug_events[0]["Status"] = "Completed"
                debug_events[0]["Details"] = (
                    f"Intent: {plan.get('intent', 'coverage')}; "
                    f"insurer: {insurer_key or 'none'}; "
                    f"policy: {policy or 'none'}"
                )
                debug_events.append(
                    {
                        "Stage": "Agent plan",
                        "Status": "Completed",
                        "Duration (ms)": None,
                        "Details": " → ".join(
                            step["action"] for step in plan.get("steps", [])
                        )
                        or plan.get("intent", "coverage"),
                    }
                )
                debug_events.append(
                    {
                        "Stage": "Evidence check",
                        "Status": (
                            "Completed" if verification.get("verified") else "Warning"
                        ),
                        "Duration (ms)": None,
                        "Details": (
                            f"{result.get('retrieved', 0)} clauses; "
                            f"unsupported: {', '.join(verification.get('unsupported') or []) or 'none'}"
                        ),
                    }
                )
                answer = result["answer"]
                model_metrics = result.get("model_metrics", {})
                debug_events.append(
                    {
                        "Stage": "Answer generation",
                        "Status": (
                            "Completed" if result.get("safe") else "Warning"
                        ),
                        "Duration (ms)": answer_duration_ms,
                        "Details": (
                            f"{model_metrics.get('model', generator.model)} via "
                            f"{model_metrics.get('backend', 'ollama')}"
                        ),
                    }
                )
                total_duration_ms = round(
                    (time.perf_counter() - request_started) * 1000,
                    2,
                )
                debug_events.append(
                    {
                        "Stage": "Total request",
                        "Status": "Completed",
                        "Duration (ms)": total_duration_ms,
                        "Details": "Plan, retrieval, generation, verification",
                    }
                )
                publish_debug(
                    {
                        "stage": "Request complete",
                        "events": debug_events,
                        "runtime": get_ollama_runtime(generator.model),
                        "retrieval": [],
                        "model_metrics": model_metrics,
                    }
                )

                st.markdown(answer)
                if verification.get("unsupported"):
                    st.caption(
                        "Unverified figures: "
                        + ", ".join(verification["unsupported"])
                    )
                if result["sources"]:
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.caption(source_label(source))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get("sources", []),
                }
            )

# Chat input lives outside the content container so it stays pinned to the
# bottom of the page and every new answer appears above it.
if tool_mode == "Ask Question":
    if prompt := st.chat_input(
        "Ask about a health or motor plan, or compare two policies..."
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

# Footer
st.divider()
st.caption("BeemaSathi v0.1 | Built with ❤️ in India | Not affiliated with any insurer")
