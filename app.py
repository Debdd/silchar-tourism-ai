import streamlit as st
import os

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

st.title("🌴 Silchar Tourism AI Guide")
st.markdown("### Hybrid Travel Assistant for Silchar & Barak Valley")

# --- 3. THE FULL KNOWLEDGE BASE ---
silchar_data = [
    # --- CACHAR DISTRICT (Silchar & Surroundings) ---
    "Bhuban Mahadev Temple: Located 50km from Silchar on Bhuban Hill. Built by Kachari King Lakshmi Chandra. It is the most celebrated Shiva temple in South Assam, featuring idols of Shiva and Parvati carved from solid rock. Thousands of 'Shivayats' trek 17km uphill during Mahashivaratri.",
    "Kancha Kanti Devi Temple (Udharbond): 15km from Silchar. Dedicated to an amalgamation of Goddess Durga and Kali. Built in 1806 by the Kachari King. Historically, it was known for rituals until 1818; the modern structure was built in 1978. It symbolizes power and purity (Kancha means gold).",
    "Khaspur (Dimasa Capital): 20km from Silchar. Ruins of the 1690 AD Dimasa Kachari Kingdom. Key features: The Lion Gate (Singhadwar), the Sun Gate (Suryadwar), and the King's Temple. The architecture is a unique blend of tribal Dimasa and Hindu styles with elephant-pattern entrances.",
    "Maniharan Tunnel: Located in the Bhuvan Hills. Mythological significance: Lord Krishna is believed to have used this tunnel to visit the Tribeni River. Shrines of Rama, Lakshmana, and Hanuman are located at the entrance. Mentioned in the Mahabharata.",
    "Shani Bari (Shani Mandir): Located in Janiganj, Silchar. A bustling spiritual hub where Saturday evenings involve massive crowds offering mustard oil and blue flowers.",
    "Bhairav Bari (Malugram): Dedicated to Lord Bhairav, the protector ('Kotwal') of the city.",
    "ISKCON Silchar: Situated in Ambicapatty. A beautifully built Radha-Krishna temple known for devotional chanting and Janmashtami celebrations.",
    "Dolu Lake: A serene natural spot surrounded by tea gardens. Ideal for nature lovers and birdwatching during winter.",
    "Gandhibag Park & Shahid Minar: A green park located centrally in Silchar, home to the martyrs’ memorial of the 1961 Language Movement. Gandhibag Park offers a tranquil escape amidst the bustling urban landscape.",
    "Sri Sri Radhaballabh Ashram (Shalganga): A premier Vaishnavite spiritual center founded in 1950, known for devotional practice and social service.",
    "Sri Sri Shyamsundar Mandir (Tarapur): A historic Krishna temple renowned for Rath Yatra and Janmashtami festivals.",
    "Satsang Vihar (Anukul Thakur Ashram): A peaceful spiritual center promoting meditation and community service.",
    "Hanuman Mandir (Tulapatty): A popular shrine in the market heart of Silchar, especially crowded on Tuesdays and Saturdays.",
    "Shiv Bari (Malugram): A revered Shiva temple known for strong spiritual devotion.",
    "NIT Silchar: A premier national engineering institute located on the outskirts of Silchar.",
    "Assam University (Dargakona): A central university known for its scenic green campus.",
    "GC College: A historic institution of higher learning in the Barak Valley.",
    "Women’s College: A leading women's education center in central Silchar.",
    "Cachar College: One of Silchar’s oldest and most respected colleges.",
    "Silchar Medical College & Hospital (SMCH): The largest healthcare and medical education facility in Barak Valley.",
    "S.M. Dev Civil Hospital: The oldest government hospital in central Silchar, serving as the primary district hospital.",
    "DSA Ground: The main sports stadium of Silchar, hosting cricket and football tournaments.",
    "India Club: A prestigious social and sports club known for tennis, cricket, and community gatherings.",
    "Silchar Airport (Kumbhirgram): The main air gateway to the Barak Valley region.",
    "Silchar Railway Station: A key rail link connecting Barak Valley to the rest of India.",
    "Goldighi Mall: The primary shopping and entertainment mall in Silchar.",
    "Satindra Mohan Dev Stadium: A major sports and cultural venue in Silchar hosting regional tournaments.",
    "District Museum, Silchar: Preserves artifacts and history of Barak Valley culture, tribes, and colonial era.",
    "Language Martyrs’ Memorial Stone (Shahid Bedi): Dedicated to the 11 martyrs of the 1961 Bengali Language Movement.",
    "Kurbantilla Mosque: One of the oldest mosques in Silchar, representing the region’s multicultural heritage.",
    "Govinda Mandir (Tarapur): A serene Krishna temple known for devotional music and festivals.",
    "Bharatiya Vidya Bhavan Silchar: A cultural and educational institute promoting Indian heritage.",
    "Kalyani Sweets Area (Tarapur Market): A locally famous food hub known for mithai and snacks.",
    "Banskandi Hala Hanuman Temple: A popular spiritual destination near Silchar.",
    "Banskandi Madrasa: A historic Islamic study center of Northeast India.",
    "Srikona Army Cantonment Area Viewpoints: Known for scenic countryside landscapes.",
    "Dargakona Lake: A peaceful waterbody near Assam University surrounded by hills.",
    "Barak River Ghats (Silchar): Scenic riverside viewpoints popular for evening walks.",
    "Tarapur Railway Colony: A historic Anglo-Indian residential colony dating back to British times.",
    "Club Road Market: A bustling shopping and dining hub.",
    "Ambicapatty Market Area: A commercial center with eateries, shops, and local life.",
    "Phatak Bazar (Silchar): A busy local bazar area known for daily shopping, small eateries, and general stores.",
    "Janiganj Bazar: One of Silchar’s busiest market areas, popular for daily essentials and street snacks.",
    "Tarapur Bazar: A neighborhood market area around Tarapur with everyday shopping and food options.",
    "Malugram Bazar: A local market area serving nearby residential neighborhoods.",
    "Silchar Circuit House: A colonial-era government residence overlooking the town.",
    "Ranighat Area (Barak Riverside): A peaceful riverbank location for evening relaxation.",
    "Kabuganj Area: A rural escape with natural beauty outside Silchar town.",
    "Bhangarpar and Kalain Region: Known for countryside scenery and agricultural fields.",
    "Lakhisahar Area: A growing suburban locality with temples and community centers.",

    # --- DURGA PUJA LIST ---
    "Durga Puja in Silchar: The city's largest festival, celebrated with grand 'Pandals' and intricate lighting. Silchar is famous for 'Theme Pujas' that rival those of Kolkata. Peak festivities on Saptami, Ashtami, and Navami.",
    "Shyamananda Ashram Durga Puja: One of the oldest and most traditional Pujas in Silchar, known for its spiritual atmosphere.",
    "All India Radio (AIR) Colony Puja: Famous for its massive budget and innovative architectural themes.",
    "Public School Road Puja: A hub for some of the most competitive and artistically decorated Pandals.",
    "Mitali Sangha (Hospital Road): Renowned for using unique, eco-friendly materials to create thematic structures.",
    "Bisharjan (Immersion) at Sadarghat: On Dashami, idols are taken in processions to the Barak River for immersion.",

    # --- KARIMGANJ & HAILAKANDI DISTRICTS ---
    "Siddheshwar Shiva Temple (Badarpurghat): A sacred Shiva temple famous for the Baruni Mela holy dip.",
    "Badarpur Fort: A Mughal-era riverside fort overlooking the Barak River.",
    "Madan Mohan Akhra (Karimganj Town): A major Vaishnavite pilgrimage site.",
    "Ramakrishna Mission (Karimganj): A peaceful spiritual and social service center.",
    "Longai River Banks: A scenic relaxation spot in Karimganj town.",
    "Siddeshwar Bari Shiv Mandir (Hailakandi): A peaceful hillside temple ideal for meditation.",
    "Pach Pirr Mukam: A sacred site honoring five revered saints symbolizing religious harmony.",
    "Sonbeel: The largest wetland in Northeast India, famed for stunning sunset reflections.",

    # --- TEA GARDENS ---
    "Rosekandy Tea Estate: A popular historic tea garden contributing to Barak Valley’s tea identity.",
    "Udharbond Tea Estate: A picturesque tea garden near Silchar, popular for drives.",
    "Urrunabund Tea Estate: One of the most scenic estates near Silchar, surrounded by rolling green hills.",
    "Iringmara Tea Estate: A lush plantation area near Dwarbund showing traditional tea garden lifestyles.",
    "Borojalengha Tea Estate: A historic garden known for strong Assam CTC tea production.",
    "West Jalinga Tea Estate: A major estate featuring picturesque estates and green valleys.",
    "Lallamookh Tea Estate (Hailakandi): A major garden offering sweeping views of endless tea fields.",
    "Baithakhal Tea Estate (Karimganj): Famous tea estate known for its picturesque landscapes.",
    
    # --- TRAVEL TIPS ---
    "Best Time to Visit: November to February for pleasant weather and festivals.",
    "Monsoon Advisory: Trekking to hill temples like Bhuban Pahar is difficult June–August.",
    "Local Food Must-Try: Shilaer Shondesh, local fish curries, and Barak Valley tea."
]

# --- 4. CLASSIFICATION & ROUTING LOGIC ---
def classify_entry(text):
    t = text.lower()
    if any(x in t for x in ["temple", "mandir", "bari", "ashram", "mosque", "mukam", "akhra", "iskcon"]): return "Religious"
    if any(x in t for x in ["tea garden", "tea estate", "plantation"]): return "Tea Tourism"
    if any(x in t for x in ["lake", "river", "wetland", "park", "hills", "viewpoint"]): return "Nature"
    if any(x in t for x in ["ruins", "fort", "museum", "historic", "memorial", "ruin"]): return "History"
    if any(x in t for x in ["college", "university", "nit", "hospital", "medical", "madrasa"]): return "Institutional"
    if any(x in t for x in ["mall", "bazar", "market", "stadium", "club", "airport", "railway"]): return "City Life"
    if any(x in t for x in ["puja", "pandal", "bisharjan"]): return "Festivals"
    return "Travel Tips"

# Mapping for user keywords
CATEGORY_TRIGGERS = {
    "lake": "Nature", "lakes": "Nature",
    "temple": "Religious", "temples": "Religious",
    "tea": "Tea Tourism", "gardens": "Tea Tourism", "garden": "Tea Tourism",
    "history": "History", "historic": "History", "ruins": "History",
    "education": "Institutional", "hospital": "Institutional", "colleges": "Institutional",
    "shopping": "City Life", "market": "City Life", "bazar": "City Life",
    "puja": "Festivals", "festivals": "Festivals"
}

# --- 5. RAG ENGINE SETUP ---
docs = [Document(page_content=e, metadata={"category": classify_entry(e)}) for e in silchar_data]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are the Silchar Tourism Assistant. Answer ONLY using the provided context. If the answer is not in the context, say you don't know yet. Context: {context}"),
    ("human", "{input}"),
])
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.info("💡 Tip: Type 'lakes', 'tea', or 'temples' for a full list.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- Process Input ---
if user_input := st.chat_input("Ask about Silchar..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        normalized = user_input.lower().strip()

        # 1. DEFINE GENERAL INFORMATION TRIGGERS
        general_triggers = [
            'everything', 'everything about silchar', 'all info about silchar', 
            'tell me about silchar', 'what can you tell me about silchar', 
            'show me all about silchar', 'all about silchar', 
            'complete information about silchar', 'full details about silchar', 
            'silchar information', 'silchar details', 'silchar info', 
            'silchar', 'all', 'complete', 'full', 'places to visit'
        ]

        # 2. CHECK FOR GENERAL INFORMATION REQUEST
        if normalized in general_triggers:
            response = "### 🌴 Complete Silchar & Barak Valley Database\n"
            response += "Here is everything I have in my records regarding places to visit, history, and local tips:\n\n"
            
            # Displaying the entire list organized by their classified types
            for category in ["Religious", "Tea Tourism", "Nature", "History", "Institutional", "City Life", "Festivals", "Travel Tips"]:
                matches = [e for e in silchar_data if classify_entry(e) == category]
                if matches:
                    response += f"**{category}**\n"
                    response += "\n".join([f"- {m}" for m in matches]) + "\n\n"
        
        # 3. KEYWORD ROUTING (Existing Category logic)
        elif normalized in CATEGORY_TRIGGERS:
            target = CATEGORY_TRIGGERS[normalized]
            matches = [e for e in silchar_data if classify_entry(e) == target]
            response = f"### 📍 Complete {target} List\n" + "\n".join([f"- {m}" for m in matches])
        
        # 4. SEMANTIC RAG (AI response)
        else:
            with st.spinner("Analyzing your request..."):
                res = rag_chain.invoke({"input": user_input})
                response = res["answer"]
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})