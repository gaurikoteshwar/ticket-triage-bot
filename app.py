import streamlit as st
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load sample tickets
with open("tickets.json", "r") as f:
    tickets_data = json.load(f)

st.set_page_config(page_title="Customer Support Ticket Triage Bot", page_icon="🤖")
st.title("🤖 Customer Support Ticket Triage Bot (Llama 2)")
st.write("Enter a support ticket, and the AI will predict category, urgency, and suggest a response.")

# Load Llama 2 model & tokenizer (7B)
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

tokenizer, model = load_model()

# Text input for new ticket
ticket_input = st.text_area("Type your customer ticket here:")

if st.button("Analyze Ticket"):
    if not ticket_input.strip():
        st.warning("Please enter a ticket first!")
    else:
        # Build prompt
        prompt = f"""
You are a helpful customer support assistant. For the ticket below:
1. Classify the category: one of [Billing, Technical, Account, Shipping, Other]
2. Predict urgency: High, Medium, Low
3. Suggest a short, professional, empathetic response

Output ONLY in JSON format.

Ticket: "{ticket_input}"
"""

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.2
            )

        # Decode and display
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("AI Output:")
        st.code(response_text, language="json")

# Optional: display some example tickets
with st.expander("See Sample Tickets"):
    for t in tickets_data[:5]:
        st.markdown(f"**Ticket:** {t['ticket']}")
        st.markdown(f"**Category:** {t['category']} | **Urgency:** {t['urgency']}")
        st.markdown(f"**Sample Response:** {t['sample_response']}")
        st.markdown("---")