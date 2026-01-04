import streamlit as st
import os
import re

# Base LangChain and Google Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Imports from the CLASSIC namespace
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Missing Google API Key! Please add it to your .env file or Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# --- 2. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Silchar Tourism AI", page_icon="🌴", layout="wide")

# Custom CSS for a better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌴 Silchar Tourism AI Guide")
st.markdown("Your specialized assistant for Silchar, Cachar, and the Barak Valley.")

# --- 3. KNOWLEDGE BASE (Full Dataset) ---
silchar_data = [
    # DISTRICT & GENERAL INFO
    "Silchar: Known as the 'Island of Peace', it is the gateway to Barak Valley. Founded in 1832. Famous for its tea industry and the 1961 Language Movement.",
    "Cachar District: The administrative district where Silchar is located, bordered by Manipur and Bangladesh.",
    
    # RELIGIOUS SITES
    "Bhuban Mahadev Temple: Located 50km from Silchar on Bhuban Hill. Features rock-cut idols of Shiva and Parvati. A 17km trek is famous during Mahashivaratri.",
    "Kancha Kanti Devi Temple (Udharbond): 15km from Silchar. Union of Goddess Kali and Durga. Built in 1806 by the Kachari King.",
    "ISKCON Silchar: Located in Ambicapatty. A major spiritual hub for Krishna devotees.",
    "Maniharan Tunnel: Bhuvan Hills. Mythology says Lord Krishna used this tunnel. Includes shrines for Rama, Lakshmana, and Hanuman.",
    "Siddheshwar Shiva Temple: Ancient temple at Badarpurghat on the Barak River banks.",
    "Shani Bari: Located in Janiganj, Silchar. Known for massive Saturday evening gatherings.",
    "Bhairav Bari: Located in Malugram; the deity is considered the protector of Silchar.",
    "Sri Sri Radhaballabh Ashram: Located in Shalganga, a peaceful Vaishnavite center.",
    
    # TEA GARDENS
    "Dolu Lake & Tea Estate: A beautiful lake surrounded by tea gardens. Famous for scenery and birdwatching.",
    "Udharbond Tea Estate: Picturesque plantation near Silchar city.",
    "Rosekandy Tea Estate: One of the most historic and largest tea gardens in the Cachar district.",
    "Lallamookh Tea Estate: A major tea garden in Hailakandi district with vast rolling hills.",
    
    # HISTORICAL SITES
    "Khaspur: 20km from Silchar. Ruins of the Dimasa Kachari Kingdom. See the Lion Gate (Singhadwar) and Sun Gate.",
    "Badarpur Fort: A Mughal-era fort in Karimganj district overlooking the Barak River.",
    "Language Martyrs’ Memorial (Shahid Minar): Located at Silchar Railway Station and Gandhibag, honoring the 11 martyrs of 1961.",
    
    # FESTIVALS
    "Durga Puja in Silchar: The biggest festival. Famous pandals include Mitali Sangha, Public School Road, and AIR Colony.",
    "Baruni Mela: A holy fair at Siddheshwar Shiva Temple where devotees take a dip in the Barak River.",
    
    # EDUCATION & HEALTH
    "NIT Silchar: A premier engineering institute with a sprawling green campus and multiple lakes.",
    "Assam University: Central university located at Dargakona.",
    "Silchar Medical College & Hospital (SMCH): The primary medical hub for the entire Barak Valley.",
    
    # TRANSPORT & SHOPPING
    "Silchar Airport (Kumbhirgram): The main air link to the region.",
    "Goldighi Mall: The primary modern shopping destination in Silchar town.",
    "Phatak Bazar: A historic and busy market area for local goods and food."
]

# --- 4. CATEGORY LOGIC (For direct filtering) ---
categories = {
    "temple": ["Bhuban Mahadev", "Kancha Kanti", "ISKCON", "Shani Bari", "Bhairav Bari", "Siddheshwar"],
    "tea": ["Dolu Lake", "Udharbond Tea Estate", "Rosekandy", "Lallamookh"],
    "historical": ["Khaspur", "Badarpur Fort", "Shahid Minar"],
    "education": ["NIT Silchar", "Assam University"]
}

# --- 5. RAG ENGINE ---
docs = [Document(page_content=t, metadata={"source": "Local Guide"}) for t in silchar_data]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

system_prompt = (
    "You are the Silchar Tourism AI. Use the context to answer questions. "
    "If the location isn't in the context, say it's not in your database yet. "
    "Be polite and use bullet points for lists.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build the Chain
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about Silchar..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Check for category keywords first
        keyword = user_input.lower()
        found_cat = next((k for k in categories if k in keyword), None)
        
        if found_cat:
            response_text = f"Here are some popular {found_cat} spots in Silchar:\n"
            response_text += "\n".join([f"- {item}" for item in categories[found_cat]])
            st.markdown(response_text)
        else:
            # Fallback to RAG
            with st.spinner("Searching records..."):
                res = rag_chain.invoke({"input": user_input})
                response_text = res["answer"]
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})