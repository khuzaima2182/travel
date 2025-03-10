import streamlit as st  # type: ignore
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai


st.set_page_config(page_title="Travel Assistant Chatbot", page_icon="✈️")

# Set up API key securely
os.environ["GOOGLE_API_KEY"] = "AIzaSyDcfQFFM_PEZxJTxH5KmoZANch0qMvZ2VE"  
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

Settings.llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

# Directory for storing index
storage_dir = "storage"
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

try:
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
except Exception as e:
    st.warning(f"📂 No existing index found. Creating a new one... (Error: {e})")

    documents = SimpleDirectoryReader("Data").load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)

    index.storage_context.persist(persist_dir=storage_dir)
    st.success("✅ Index created and saved successfully!")

# Improved Prompt
prompt_template = """
You are an expert AI travel assistant and Itinerary Generator, providing detailed, accurate, and well-researched responses.
Your responses must be:
- Highly informative: Provide in-depth travel insights based on real-world data.
- Context-aware: Adapt responses based on the user’s travel preferences and group type.
- Engaging but precise: Maintain a friendly tone while ensuring factual accuracy.

Important: If a user asks about a specific destination, provide detailed recommendations, including attractions, local tips, and safety advice.

Now, let's assist the traveler with an expert-level response!
"""

query_engine = index.as_query_engine(llm=Settings.llm, system_prompt=prompt_template, stream_response=True)

st.title("Concierge AI Travel Agent")
st.subheader("Hey there, traveler! Ready to plan your next adventure?")

st.sidebar.title("✨ Customize Your Trip")

# Travel Companion
travel_companion = st.sidebar.radio("Who are you traveling with?", ["Solo", "With Partner", "With Family", "With Friends"])

# Travel Interests
travel_interests = st.sidebar.multiselect(
    "What type of trip interests you?", 
    ["Nature & Wildlife", "Romantic Getaway", "Historical & Cultural", "Adventure & Thrill", "Food & Culinary", "Luxury & Relaxation"]
)

st.sidebar.markdown("---")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask me anything about your trip! ✈️")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_container = st.empty()  
        st.markdown("...")  

        with st.spinner("Thinking... 🤔"):
            
            travel_context = f"You are traveling {travel_companion.lower()} and interested in {', '.join(travel_interests)}." if travel_interests else "No specific preferences given."
            full_query = f"Context: {travel_context}\n\nUser Question: {user_input}"

            try:
                response_stream = query_engine.query(full_query)
                bot_reply = ""

               
                if hasattr(response_stream, "response_gen"):
                    for chunk in response_stream.response_gen:
                        bot_reply += chunk
                        response_container.markdown(bot_reply)
                else:
                    bot_reply = response_stream.response
                    response_container.markdown(bot_reply)

                #
                bot_reply += "\n\n🌟 Let me know if you need more details or a different suggestion!"
            except Exception as e:
                bot_reply = f"❌ Oops, something went wrong: {e}"
                response_container.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})


if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
