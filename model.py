import streamlit as st  # type: ignore
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai

# ✅ Streamlit page config (must be the first command)
st.set_page_config(page_title="Travel Assistant Chatbot", page_icon="✈️")

# ✅ Configure Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDcfQFFM_PEZxJTxH5KmoZANch0qMvZ2VE"  # Replace with your actual API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Initialize LlamaIndex with Gemini
Settings.llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

# ✅ Define storage path
storage_dir = r'C:\Users\user\Desktop\New folder\storage'

# ✅ Ensure storage directory exists
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

try:
    # ✅ Attempt to Load Existing Index
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
except Exception as e:
    # ✅ Handle missing index scenario
    st.warning(f"📂 No existing index found. Creating a new one... (Error: {e})")

    # ✅ Load documents and create new index
    documents = SimpleDirectoryReader(r'C:\Users\user\Desktop\New folder\Data').load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)

    # ✅ Save the index
    index.storage_context.persist(persist_dir=storage_dir)
    st.success("✅ Index created and saved successfully!")

# ✅ Updated System Prompt for a More Humanized Tone
prompt_template = """
You are a friendly and enthusiastic AI-powered travel guide, eager to make every trip feel exciting and memorable! 
Your responses should be warm, engaging, and natural—just like a conversation with a well-traveled friend.

How you should interact:
- Greet users warmly and make them feel special.
- Keep the tone friendly, cheerful, and full of positive energy.
- Offer thoughtful travel recommendations based on their preferences.
- Use casual, inviting language while keeping responses informative.
- Add a touch of excitement and storytelling to inspire travelers.

Now, let's help our traveler!   
"""

query_engine = index.as_query_engine(llm=Settings.llm, system_prompt=prompt_template, stream_response=True)

# ✅ Streamlit UI
st.title("🌍 AI Travel Agent")
st.subheader("Hey there, traveler! Ready to plan your next adventure? ✨")

# ✅ Sidebar for Travel Preferences
st.sidebar.title("✨ Customize Your Trip")

# Travel Companion
travel_companion = st.sidebar.radio("Who are you traveling with?", 
                                    ["Solo", "With Partner", "With Family", "With Friends"])

# Travel Interests
travel_interests = st.sidebar.multiselect(
    "What type of trip interests you?", 
    ["Nature & Wildlife", "Romantic Getaway", "Historical & Cultural", "Adventure & Thrill", "Food & Culinary", "Luxury & Relaxation"]
)

st.sidebar.markdown("---")

# ✅ Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Input
user_input = st.chat_input("Ask me anything about your trip! ✈️")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        st.markdown("Just a sec... ✨ Let me put together something amazing for you!")

        # ✅ Stream AI response
        with st.spinner("🌍 Mapping out the best recommendations for you..."):
            travel_context = f"You are traveling {travel_companion.lower()} and looking for {', '.join(travel_interests)} experiences." if travel_interests else ""
            full_query = f"{travel_context} {user_input}"

            try:
                response_stream = query_engine.query(full_query)
                response_container = st.empty()  # Placeholder for dynamic updates
                
                bot_reply = ""
                if hasattr(response_stream, "response_gen"):  # Handle streaming responses
                    for chunk in response_stream.response_gen:
                        bot_reply += chunk
                        response_container.markdown(bot_reply)
                else:  # Handle non-streaming responses
                    bot_reply = response_stream.response
                    response_container.markdown(bot_reply)

                # ✅ Add a final warm touch to the response
                bot_reply += "\n\n🌟 I hope this helps! Let me know if you'd like more details or a different suggestion."

            except Exception as e:
                bot_reply = f"❌ Oops, something went wrong: {e}"
                response_container.markdown(bot_reply)

    # ✅ Store response in chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# ✅ Clear Chat Button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
