import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = st.sidebar.text_input(
    "API URL",
    "http://localhost:8001",
)

API_KEY = os.getenv("API_KEY", "")


st.title("Eligibility Pre-Qualification")
st.caption(
    "Preliminary assessment based on authoritative eligibility rules."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def ask_api(prompt):
    try:
        response = requests.post(
            f"{API_URL}/ask",
            headers={
                "Accept": "application/json",
                "X-API-Key": API_KEY,
            },
            json={
                "question": prompt,
            },
            timeout=180,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to API at {API_URL}. "
            "Make sure the FastAPI server is running."
        )
        return None

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")
        return None

    except requests.exceptions.HTTPError as exc:
        st.error(
            f"API returned an error: "
            f"{exc.response.status_code} - {exc.response.text}"
        )
        return None


prompt = st.chat_input(
    "Enter applicant information..."
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("LoanAssist is preparing your response..."):
            result = ask_api(prompt)

        if result:
            answer = result.get("answer", "")
            citations = result.get("citations", [])

            if answer:
                st.markdown(answer)
            else:
                st.warning("The API returned no answer.")

            if citations:
                st.markdown("### Sources")

                for i, citation in enumerate(citations, start=1):
                    doc = citation.get("doc", "Unknown document")
                    text = citation.get("text", "")

                    with st.expander(f"{i}. {doc}"):
                        st.markdown(text)

            if result.get("refused"):
                st.warning(
                    f"Request refused: {result.get('reason', 'No reason provided')}"
                )

            assistant_content = answer
            if citations:
                assistant_content += f"\n\n### Sources\n{citations}"
        else:
            assistant_content = ""

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_content,
        }
    )
