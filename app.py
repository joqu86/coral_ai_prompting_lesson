import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Class Lab", page_icon="🤖", layout="centered")

MODELS = {
    "GPT-4o mini (fast, cheap)": "gpt-4o-mini",
    "GPT-4o (stronger)": "gpt-4o",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1": "gpt-4.1",
}

SYSTEM_PROMPT = (
    "You are a helpful tutor for a workplace AI class. "
    "Give clear, concise answers. If asked for harmful or non-educational "
    "content, refuse politely."
)

STARTER_PROMPTS = [
    "Explain what a large language model is in simple terms.",
    "What makes a good prompt? Give three tips.",
    "Critique this draft email: Hi team, just checking in on the thing.",
]


def require_auth() -> None:
    if st.session_state.get("authenticated"):
        return

    st.title("AI Class Lab")
    st.caption("Enter the class password to continue.")

    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Sign in", type="primary"):
        expected = st.secrets.get("APP_PASSWORD")
        if not expected:
            st.error("App is not configured: set APP_PASSWORD in Streamlit secrets.")
            st.stop()
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")

    st.stop()


def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("App is not configured: set OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


def run_turn(
    client: OpenAI,
    model_id: str,
    model_label: str,
    temperature: float,
    prompt: str,
) -> str:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model_label}…"):
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                ],
                temperature=temperature,
            )
            answer = response.choices[0].message.content or ""
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer


def render_chat(model_id: str, model_label: str, temperature: float) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_prompt", None)
    if pending:
        run_turn(get_client(), model_id, model_label, temperature, pending)
        return

    if prompt := st.chat_input("Ask something…"):
        run_turn(get_client(), model_id, model_label, temperature, prompt)


def main() -> None:
    require_auth()

    st.title("AI Class Lab")
    st.caption(
        "Try different models and compare answers. "
        "Do not paste passwords, grades, or other private data."
    )

    with st.sidebar:
        st.subheader("Settings")
        model_label = st.selectbox("Model", list(MODELS.keys()))
        model_id = MODELS[model_label]
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.1)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Sign out"):
            st.session_state.authenticated = False
            st.session_state.pop("messages", None)
            st.rerun()

        st.divider()
        st.subheader("Starter prompts")
        for i, text in enumerate(STARTER_PROMPTS):
            if st.button(text, key=f"starter_{i}"):
                st.session_state.messages = []
                st.session_state.pending_prompt = text
                st.rerun()

    render_chat(model_id, model_label, temperature)


main()
