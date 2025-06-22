import streamlit as st
import base64
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials

st.set_page_config(layout="wide")
st.title("📄 Invoice Analyzer with Chat")

# --- Helper Function ---
def encode_uploaded_image(uploaded_file):
    """Encodes a Streamlit UploadedFile object to a base64 string."""
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

# --- Model Information ---
MODELS = [
    "ibm/granite-vision-3-2-2b",
    "meta-llama/llama-3-2-11b-vision-instruct",
    "meta-llama/llama-3-2-90b-vision-instruct",
    "meta-llama/llama-guard-3-11b-vision",
    "mistralai/pixtral-12b"
]

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "encoded_image" not in st.session_state:
    st.session_state.encoded_image = ""
if "initial_analysis_done" not in st.session_state:
    st.session_state.initial_analysis_done = False

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Select a Model", MODELS)
    api_key = st.text_input("Enter your watsonx.ai API Key", type="password")
    project_id = st.text_input("Enter your watsonx.ai Project ID")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload an invoice image", type=["jpg", "png", "jpeg"])

    if st.button("Analyze Invoice", use_container_width=True, disabled=not uploaded_file):
        if not api_key or not project_id:
            st.error("API Key and Project ID are required.")
        else:
            with st.spinner("Analyzing invoice... This may take a moment."):
                # Reset state for new analysis
                st.session_state.messages = []
                st.session_state.initial_analysis_done = False
                st.session_state.encoded_image = encode_uploaded_image(uploaded_file)

                try:
                    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
                    params = TextChatParameters(max_tokens=2048, temperature=0.0)
                    model = ModelInference(
                        model_id=selected_model,
                        credentials=credentials,
                        project_id=project_id,
                        params=params
                    )
                    initial_prompt = (
                        "Extract all relevant product and invoice information from this image and return it as a well-structured JSON object. "
                        "Group items by category (based on the product description blocks)."
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": initial_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{st.session_state.encoded_image}"}},
                            ],
                        }
                    ]
                    response_obj = model.chat(messages=messages)
                    if response_obj and response_obj["choices"]:
                        response = response_obj["choices"][0]["message"]["content"]
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.initial_analysis_done = True
                        st.rerun()
                    else:
                        st.error("Failed to get a valid response from the model.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- Main Content Area ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Uploaded Invoice")
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
    else:
        st.info("Upload an invoice and click 'Analyze Invoice' in the sidebar to begin.")

with col2:
    st.subheader("Analysis & Chat")
    if not st.session_state.initial_analysis_done:
        st.info("Analysis results and chat will appear here after you analyze an invoice.")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for follow-up questions
    if st.session_state.initial_analysis_done:
        if prompt := st.chat_input("Ask a follow-up question about the invoice..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                try:
                    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
                    params = TextChatParameters(max_tokens=2048, temperature=0.0)
                    model = ModelInference(
                        model_id=selected_model,
                        credentials=credentials,
                        project_id=project_id,
                        params=params
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{st.session_state.encoded_image}"}},
                            ],
                        }
                    ]
                    response_obj = model.chat(messages=messages)
                    if response_obj and response_obj["choices"]:
                        response = response_obj["choices"][0]["message"]["content"]
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get a valid response from the model.")

                except Exception as e:
                    st.error(f"An error occurred during chat: {e}")
