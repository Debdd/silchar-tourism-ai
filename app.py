import streamlit as st
import os

# Base LangChain and Google Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- THE FIX: Import from the CLASSIC namespace ---
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv  # Import dotenv

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # This looks for a .env file and loads its variables
google_api_key = os.getenv("GOOGLE_API_KEY")

GENERAL_SILCHAR_INFO = """
Silchar, the 'Island of Peace,' is the headquarters of Cachar district and the gateway to the Barak Valley. 
- Name Origin: Derived from 'Shil' (Rock) and 'Char' (Riverside), referring to the rocky banks of the Barak River.
- History: Founded in 1832 by Captain Thomas Fisher; it is home to the world's first polo club (1850).
- Language: Primarily Bengali (Sylheti dialect), but English, Hindi, and Assamese are widely understood.
- Geography: Surrounded by Manipur (East), Bangladesh (West), Mizoram (South), and the Barail Hills (North).
- Best Time to Visit: November to February (Cool and pleasant).
- Famous For: Tea gardens, paper mills (HPC), and historic Kachari kingdom ruins.
"""

# Safety check: if the key is missing from .env, show a warning
if not google_api_key:
    st.error("Missing Google API Key! Please add it to your .env file.")
    st.stop()

# Set the key for LangChain
os.environ["GOOGLE_API_KEY"] = google_api_key


# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Silchar Tourism Guide", page_icon="🌴")
st.title("🌴 Silchar Tourism AI Guide")
st.markdown("Your personal travel assistant for the Barak Valley, Assam.")

# Sidebar for API Key
#google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # --- 2. KNOWLEDGE BASE (Your Data) ---
    # In a real project, you can load this from a .txt or .pdf file
    silchar_data = [
        # --- CACHAR DISTRICT (Silchar & Surroundings) ---
        "Bhuban Mahadev Temple: Located 50km from Silchar on Bhuban Hill. Built by Kachari King Lakshmi Chandra. It is the most celebrated Shiva temple in South Assam, featuring idols of Shiva and Parvati carved from solid rock. Thousands of 'Shivayats' trek 17km uphill during Mahashivaratri.",
        "Kancha Kanti Devi Temple (Udharbond): 15km from Silchar. Dedicated to an amalgamation of Goddess Durga and Kali. Built in 1806 by the Kachari King. Historically, it was known for rituals until 1818; the modern structure was built in 1978. It symbolizes power and purity (Kancha means gold).",
        "Khaspur (Dimasa Capital): 20km from Silchar. Ruins of the 1690 AD Dimasa Kachari Kingdom. Key features: The Lion Gate (Singhadwar), the Sun Gate (Suryadwar), and the King's Temple. The architecture is a unique blend of tribal Dimasa and Hindu styles with elephant-pattern entrances.",
        "Maniharan Tunnel: Located in the Bhuvan Hills. Mythological significance: Lord Krishna is believed to have used this tunnel to visit the Tribeni River. Shrines of Rama, Lakshmana, and Hanuman are located at the entrance. Mentioned in the Mahabharata.",
        "Shani Bari (Shani Mandir): Located in Janiganj, Silchar. A bustling spiritual hub where Saturday evenings involve massive crowds offering mustard oil and blue flowers. It is the heartbeat of Silchar's local religious life.",
        "Bhairav Bari (Malugram): Dedicated to Lord Bhairav, the protector ('Kotwal') of the city. Locals visit here before starting new ventures to seek protection.",
        "ISKCON Silchar: Situated in Ambicapatty. One of the best-built temples in the region, featuring exquisite marble idols of Radha-Krishna and hosting vibrant Janmashtami celebrations.",
        "Dolu Lake: A serene natural spot surrounded by lush tea gardens. Ideal for nature lovers and birdwatching during the winter months.",
        "Gandhibag Park: Located in the city center. It houses the Shahid Minar, a memorial for the 11 martyrs of the 1961 Language Movement, making it a site of both leisure and historical pride.",
        "Sri Sri Radhaballabh Ashram (Shalganga): Located near Silchar Airport (Kumbhirgram). Founded by Prabhupad Sri Sri Braja Raman Goswami in 1950. It is a premier center of Vaishnavite culture and humanitarian service in the Barak Valley. he main temple houses Sri Sri Radha Ballabh, flanked by Sri Sri Durga Mata and Sri Sri Katyayanee Mata. It is also home to the sacred 'Jugal Kadamba' tree at Kalachand Tala, representing Sri Sri Rai Kalachand.",
        "Sri Sri Shyamsundar Mandir (Tarapur): Located in Tula Patty, Tarapur, Silchar. It is one of the oldest and most revered Vaishnavite temples in the city, dedicated to Lord Shyamsundar (Krishna) and Radha. Famous for its grand celebration of the Rath Yatra (Festival of Chariots), where beautifully decorated chariots carry the deities through the streets of Silchar. It also hosts significant festivities during Janmashtami, Jhulan Yatra, and Dol Jatra (Holi).",
        "Satsang Vihar (Anukul Thakur Ashram): A spiritual center dedicated to Sri Sri Thakur Anukulchandra, known for its peaceful environment and community service.",
        "Hanuman Mandir (Tulapatty): Located on Tulapatty Road in the heart of Silchar's commercial hub. It is a vibrant community shrine dedicated to Lord Hanuman, serving as a spiritual anchor for local traders and residents. The temple is especially crowded on Tuesdays and Saturdays, known for its traditional morning and evening aartis that resonate through the busy market streets.",
        "Shiv Bari (Malugram): A historic and highly revered Shiva temple in the Malugram area of Silchar, known for its spiritual significance and vibrant celebrations during Mahashivratri.",
        "Silchar Airport (Kumbhirgram): The primary gateway to the Barak Valley, located about 22km from the city.",
        "Silchar Railway Station: A major railhead connecting the region to the rest of India, located in the Tarapur area.",
        "Goldighi Mall: The city's prominent shopping destination located in the heart of Silchar.",
        "District Library: A key cultural and educational center for the community.",
        "DSA Ground: The main sports stadium in Silchar, hosting cricket, football, and major local events.",
        "India Club: One of the oldest and most prestigious social clubs in the city.",
        "NIT Silchar: A premier technical institute of national importance located on the outskirts of the city.",
        "GC College: A historic educational institution known for its academic heritage in the Barak Valley.",
        "Silchar Medical College and Hospital (SMCH): The premier healthcare and medical education facility in the region.",
        "Assam University: A central university located at Dargakona, known for its sprawling campus and diverse academic programs.",
        "Women's College: A premier institution for women's education in Silchar, centrally located near Club Road.",
        "Cachar College: One of the oldest colleges in the city, situated in the Trunk Road area and known for its historic academic standing."
    
        # --- KARIMGANJ DISTRICT ---

        "Siddheshwar Shiva Temple (Badarpurghat): An ancient temple on the banks of the Barak River. It is a major center for the 'Baruni Mela' festival where thousands take a holy dip in the river.",
        "Badarpur Fort: A Mughal-era fort situated on the banks of the Barak. It was a strategic military point and now serves as a key historical ruin for tourists.",
        "Kal-Bhairab Bari (Karimganj): A sacred site in Banamali, Karimganj. Known for its fierce manifestation of Shiva and visited by those seeking spiritual strength.",
        "Madan Mohan Akhra: Located in Karimganj Town. A premier Vaishnavite pilgrimage site where Lord Krishna is worshipped with traditional 'Kirtans'.",
        "Ramakrishna Mission (Karimganj): A spiritual retreat and social service center following the teachings of Swami Vivekananda, known for its calm and meditative environment.",

        # --- HAILAKANDI DISTRICT ---
        "Siddeshwar Bari Sibmandir: Located near the Badarpur Ghat region on the Hailakandi side. A quiet, mist-wrapped temple known for spiritual reflection during winter mornings.",
        "Pach Pirr Mukam: A sacred site on the southern side of Hailakandi dedicated to five revered saints, symbolizing the religious harmony of the Barak Valley.",
        "Sonbeel (Hailakandi/Karimganj border): The largest wetland in Northeast India. Though not a temple, it is a spiritual experience for nature lovers, famous for its 'Shako' (wooden bridges) and sunset views.",
        
        # --- LOCAL TRAVEL TIPS ---
        "Best Time to Visit: November to February is ideal for pleasant weather and festivals. Monsoon (June-August) offers lush greenery but makes hill treks like Bhuban Pahar difficult.",
        "Local Food: Tourists must try 'Shilaer Shondesh' and local fish curry. Popular eateries include Hashtag Cafe and Shakahaar."
    ]

    # --- 3. RAG ENGINE (Processing Data) ---
    # Turn text into Documents
    docs = [Document(page_content=text) for text in silchar_data]
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Database (In-Memory for this demo)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # --- 4. MODERN RETRIEVAL CHAIN ---
    # Setup the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    system_prompt = f"""
        You are a helpful Silchar Tourism Assistant. 
        1. First, search the provided 'silchar_data' for specific details.
        2. If the user asks about something NOT in 'silchar_data' (e.g., general history, weather, or location), 
           use the following GENERAL_SILCHAR_INFO to answer:
           {GENERAL_SILCHAR_INFO}
        3. If the query is completely unrelated to Silchar or tourism, politely redirect them back to Silchar topics.
        4. Always provide information in a friendly and helpful tone, suitable for tourists.
        5. Keep responses concise and easy to understand for tourists.
        6. If you don't have specific information in 'silchar_data', use the GENERAL_SILCHAR_INFO to provide helpful context.
        7. When referencing specific places or attractions, try to include nearby landmarks or transportation options.
        8. If suggesting restaurants or hotels, mention their approximate location or nearby attractions.
        9. When providing directions or travel tips, keep them practical and easy to follow.
        10. Always end with a friendly suggestion to explore more of Silchar!
        11. If the user seems interested in a particular topic, suggest related places or activities in Silchar.
        12. If the user asks for recommendations, provide at least 2-3 options with brief descriptions.
        13. When discussing local culture or events, try to connect them to popular tourist attractions.
        14. If discussing festivals or events, mention nearby places to visit or eat during those times.
        15. When providing information about local cuisine, suggest nearby restaurants or food stalls.
        16. If discussing local markets or shopping, mention nearby attractions or dining options.
        17. When discussing transportation, mention nearby bus stops, taxi services, or ride-sharing options.
        18. If discussing accommodation, suggest nearby attractions or amenities.
        19. When providing information about local events or activities, suggest nearby places to visit or eat.
        20. If the user asks about safety or travel tips, provide general advice while encouraging them to check local resources.
        21. When providing information about local festivals or celebrations, suggest nearby places to visit or experience during those times.
        22. If the user asks about local traditions or customs, connect them to nearby cultural sites or events.
        23. If the user asks about local wildlife or nature, suggest nearby parks or nature spots to explore.
        24. If the user asks about local handicrafts or souvenirs, suggest nearby markets or shops to visit.
        25. If the user asks about local transportation options, suggest nearby stations or services.
        26. If the user asks about local events or activities, suggest nearby places to visit or experience.
        27. If the user asks about local guides or tour services, suggest nearby tour operators or guide services.
        28. If the user asks about local photography spots, suggest nearby scenic locations or viewpoints.
        29. If the user asks about local festivals or celebrations, suggest nearby places to visit or experience during those times.
        30. If the user asks about local art or cultural experiences, suggest nearby galleries, museums, or cultural centers."""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
        ]
    )

    # Combine the steps into a RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 5. CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Ask Anything about Silchar's attractions and history (e.g., temples, colleges, forts, etc.) - try 'Tell me about Shani Bari' or 'What colleges are in Silchar?'"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # The new rag_chain returns a dictionary with an "answer" key
            response = rag_chain.invoke({"input": user_input})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to begin your journey through Silchar!")
