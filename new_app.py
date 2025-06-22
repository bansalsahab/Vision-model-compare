from __future__ import annotations

import base64
import io
import re
from typing import List

import streamlit as st
import pandas as pd
import time
from PIL import Image, ImageSequence
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# -----------------------------------------------------------------------------
# Constants / Benchmarks
# -----------------------------------------------------------------------------
MODELS = {
    "ibm/granite-vision-3-2-2b": {
        "description": "Granite Vision 3.2 B (IBM) — balanced vision-language model.",
        "benchmarks": {
            "MMMU (val)": 52.4,
            "MathVista (mini)": 47.1,
            "AI2D (test)": 77.1,
            "ChartQA (test)": 76.4,
            "DocVQA (val)": 81.3,
        },
    },
    "meta-llama/llama-3-2-11b-vision-instruct": {
        "description": "Llama-3 2 11 B Vision (Meta) — instruction-tuned vision model.",
        "benchmarks": {
            "MMMU (val)": 60.3,
            "MathVista (mini)": 56.4,
            "VQA-v2 (dev)": 83.1,
            "MM-Bench (test)": 79.4,
            "DocVQA (val)": 86.7,
        },
    },
    "meta-llama/llama-3-2-90b-vision-instruct": {
        "description": "Llama-3 2 90 B Vision (Meta) — high-capacity variant.",
        "benchmarks": {
            "MMMU (val)": 67.4,
            "MathVista (mini)": 69.9,
            "VQA-v2 (dev)": 86.4,
            "MM-Bench (test)": 84.7,
            "DocVQA (val)": 90.1,
        },
    },
    "meta-llama/llama-guard-3-11b-vision": {
        "description": "Llama-Guard 3 11 B Vision — safety classifier.",
        "benchmarks": {
            "VL-Safety-Bench (acc)": "97%",
        },
    },
    "mistralai/pixtral-12b": {
        "description": "Pixtral 12 B (Mistral AI) — strong general-purpose VL model.",
        "benchmarks": {
            "MMMU (val)": 63.8,
            "MathVista (mini)": 60.2,
            "VQA-v2 (dev)": 84.2,
            "MM-Bench (test)": 81.5,
            "DocVQA (val)": 88.1,
        },
    },
}
MODEL_NAMES = list(MODELS.keys())
EVAL_MODEL = "ibm/granite-3-3-8b-instruct"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _encode_image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()

def _extract_invoice(model: ModelInference, img_bytes: bytes) -> str:
    """Send vision prompt using chat() API and return the text content."""
    prompt = (
        "Extract all relevant product and invoice information from this image and "
        "return it as a well-structured JSON object. Group items by category. "
        "Include: style_no, quantity, unit, unit_price_usd, total_cost_usd, "
        "invoice_total_usd, discount_usd, final_total_usd. Return only the JSON "
        "output."
    )
    encoded = _encode_image_to_base64(img_bytes)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }
    ]
    resp = model.chat(messages=messages)
    return resp["choices"][0]["message"]["content"]

def _evaluate_outputs(creds: Credentials, project_id: str, results: List[dict]) -> str:
    """Use EVAL_MODEL to compare JSON outputs and return ranking text."""
    if not results:
        return "No results to evaluate."

    prompt_parts = [
        "You are an expert data quality analyst. Compare the following JSON outputs extracted from the same invoice by different vision-language models and decide which output is the most complete and correct. Consider: presence of all items, numeric correctness, duplicates, and presence of invoice totals. Provide a ranked list of model names from best to worst with brief reasoning."
    ]
    for res in results:
        prompt_parts.append(f"\nModel: {res['model']}\n```json\n{res['json']}\n```")

    full_prompt = "\n".join(prompt_parts)
    params = TextChatParameters(max_tokens=1024, temperature=0.0)
    model = ModelInference(model_id=EVAL_MODEL, credentials=creds, project_id=project_id, params=params)
    eval_resp = model.chat(messages=[{"role": "user", "content": full_prompt}])
    return eval_resp["choices"][0]["message"]["content"]

def convert_tif_bytes_to_png_bytes(tif_bytes: bytes) -> List[bytes]:
    out: List[bytes] = []
    with Image.open(io.BytesIO(tif_bytes)) as img:
        for page in ImageSequence.Iterator(img):
            buf = io.BytesIO()
            page.convert("RGB").save(buf, format="PNG")
            out.append(buf.getvalue())
    return out

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Vision Model Dashboard", layout="wide")
st.title("🖼️ Vision Model Dashboard")

# Sidebar – credentials & model chooser
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("watsonx.ai API key", type="password")
    project_id = st.text_input("watsonx.ai Project ID")
    st.markdown("---")
    st.caption("Upload a single invoice image (PNG/JPG/TIFF). It will be parsed by all 5 models.")
    uploaded_file = st.file_uploader("Invoice image", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    run_btn = st.button("Run extraction", disabled=uploaded_file is None)

# Main – column layout
col_left, col_right = st.columns([1, 2])

# -----------------------------------------------------------------------------
# Left column – image preview and results table
# -----------------------------------------------------------------------------
with col_left:
    st.subheader("Uploaded Invoice")
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
    else:
        st.info("Upload an invoice image to begin.")

    if "bench_df" in st.session_state:
        st.subheader("Timing Comparison (seconds)")
        st.bar_chart(st.session_state.bench_df.set_index("Model"))

# -----------------------------------------------------------------------------
# Right column – extraction results
# -----------------------------------------------------------------------------
with col_right:
    st.subheader("Extraction Results")

    if run_btn:
        if not api_key or not project_id:
            st.error("Please provide API credentials in the sidebar.")
        else:
            creds_base = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
            params_base = TextChatParameters(max_tokens=2048, temperature=0.0)
            
            st.session_state.results = []
            bench_records = []

            # prepare image bytes (handle TIFF conversion for first page only)
            if uploaded_file is not None:
                ext = uploaded_file.name.lower()
                if ext.endswith((".tif", ".tiff")):
                    img_bytes = convert_tif_bytes_to_png_bytes(uploaded_file.getvalue())[0]
                else:
                    img_bytes = uploaded_file.getvalue()
            else:
                st.error("No file uploaded")
                st.stop()

            for model_name in MODEL_NAMES:
                try:
                    start = time.perf_counter()
                    model = ModelInference(model_id=model_name, credentials=creds_base, project_id=project_id, params=params_base)
                    json_str = _extract_invoice(model, img_bytes)
                    duration = time.perf_counter() - start
                    st.session_state.results.append({"model": model_name, "json": json_str, "time": duration})
                    bench_records.append({"Model": model_name, "Time": round(duration, 2)})
                    st.success(f"{model_name} ✅ ({duration:.2f}s)")
                except Exception as e:
                    bench_records.append({"Model": model_name, "Time": None})
                    st.error(f"{model_name} ❌ {e}")

            st.session_state.bench_df = pd.DataFrame(bench_records)

            # Evaluate outputs using Granite 3.3 8B instruct
            try:
                eval_text = _evaluate_outputs(creds_base, project_id, st.session_state.results)
                st.session_state.eval_text = eval_text
            except Exception as e:
                st.session_state.eval_text = f"Evaluation failed: {e}"

    # Show detailed outputs
    if "results" in st.session_state and st.session_state.results:
        for res in st.session_state.results:
            with st.expander(res["model"], expanded=False):
                st.code(res["json"], language="json")

        # Evaluation summary
        if "eval_text" in st.session_state:
            st.subheader("Model Comparison (Granite 3.3 8B Instruct)")
            st.markdown(st.session_state.eval_text)

        # Highlight fastest model
        fast_df = st.session_state.bench_df.dropna(subset=["Time"]).sort_values("Time")
        if not fast_df.empty:
            fastest = fast_df.iloc[0]
            st.info(f"Fastest model: **{fastest['Model']}** ⏱️ {fastest['Time']} seconds")
