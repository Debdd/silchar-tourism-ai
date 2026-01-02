import streamlit as st
import os
import re

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
    "Shani Bari (Shani Mandir): Located in Janiganj, Silchar. A bustling spiritual hub where Saturday evenings involve massive crowds offering mustard oil and blue flowers.",
    "Bhairav Bari (Malugram): Dedicated to Lord Bhairav, the protector ('Kotwal') of the city.",
    "ISKCON Silchar: Situated in Ambicapatty. A beautifully built Radha-Krishna temple known for devotional chanting and Janmashtami celebrations.",
    "Dolu Lake: A serene natural spot surrounded by tea gardens. Ideal for nature lovers and birdwatching during winter.",
    "Gandhibag Park & Shahid Minar: A green park located centrally in Silchar, home to the martyrs’ memorial of the 1961 Language Movement. Gandhibag Park offers a tranquil escape amidst the bustling urban landscape. Named in honor of Mahatma Gandhi, this park is a verdant oasis that beckons visitors with its lush greenery and serene ambiance.",
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

    # --- NEW CACHAR / SILCHAR LANDMARKS ---
    "Satindra Mohan Dev Stadium: A major sports and cultural venue in Silchar hosting regional tournaments and public events.",
    "District Museum, Silchar: Preserves artifacts and history of Barak Valley culture, tribes, and colonial era.",
    "Language Martyrs’ Memorial Stone (Shahid Bedi): Dedicated to the 11 martyrs of the 1961 Bengali Language Movement.",
    "Kurbantilla Mosque: One of the oldest mosques in Silchar, representing the region’s multicultural heritage.",
    "Govinda Mandir (Tarapur): A serene Krishna temple known for devotional music and festivals.",
    "Bharatiya Vidya Bhavan Silchar: A cultural and educational institute promoting Indian heritage.",
    "Kalyani Sweets Area (Tarapur Market): A locally famous food hub known for mithai and snacks.",
    "Banskandi Hala Hanuman Temple: A popular spiritual destination near Silchar.",
    "Banskandi Madrasa (Islamic Theological Institute): A historic Islamic study center of Northeast India.",
    "Tea Gardens around Silchar: Scenic plantation belts offering rural landscape views.",
    "Udharbond Tea Estate: A picturesque tea garden near Silchar, popular for drives.",
    "Srikona Army Cantonment Area Viewpoints: Known for scenic countryside landscapes.",
    "Dargakona Lake: A peaceful waterbody near Assam University surrounded by hills.",
    "Barak River Ghats (Silchar): Scenic riverside viewpoints popular for evening walks.",
    "Tarapur Railway Colony: A historic Anglo-Indian residential colony dating back to British times.",
    "Club Road Market: A bustling shopping and dining hub.",
    "Ambicapatty Market Area: A commercial center with eateries, shops, and local life.",
    "Phatak Bazar (Silchar): A busy local bazar area known for daily shopping, small eateries, and general stores.",
    "Janiganj Bazar: One of Silchar’s busiest market areas, popular for daily essentials, street snacks, and local shopping.",
    "Tarapur Bazar: A neighborhood market area around Tarapur with everyday shopping and food options.",
    "Malugram Bazar: A local market area serving nearby residential neighborhoods with daily essentials.",
    "Silchar Circuit House: A colonial-era government residence overlooking the town.",
    "Ranighat Area (Barak Riverside): A peaceful riverbank location for evening relaxation.",
    "Kabuganj Area: A rural escape with natural beauty outside Silchar town.",
    "Bhangarpar and Kalain Region: Known for countryside scenery and agricultural fields.",
    "Lakhisahar Area: A growing suburban locality with temples and community centers.",

    # --- KARIMGANJ DISTRICT ---
    "Siddheshwar Shiva Temple (Badarpurghat): A sacred Shiva temple famous for the Baruni Mela holy dip.",
    "Badarpur Fort: A Mughal-era riverside fort overlooking the Barak River.",
    "Madan Mohan Akhra (Karimganj Town): A major Vaishnavite pilgrimage site known for devotional chanting.",
    "Ramakrishna Mission (Karimganj): A peaceful spiritual and social service center inspired by Swami Vivekananda.",
    "Longai River Banks: A scenic relaxation spot in Karimganj town.",

    # --- HAILAKANDI DISTRICT ---
    "Siddeshwar Bari Shiv Mandir: A peaceful hillside temple ideal for meditation and devotion.",
    "Pach Pirr Mukam: A sacred site honoring five revered saints symbolizing religious harmony.",
    "Sonbeel: The largest wetland in Northeast India, famed for stunning sunset reflections on the water.",

        # --- TEA GARDENS OF BARAK VALLEY (Cachar • Hailakandi • Karimganj) ---
    "Urrunabund Tea Estate (Cachar): One of the most scenic tea estates near Silchar, surrounded by rolling green hills and peaceful countryside views. A great photography spot for visitors exploring rural Barak Valley.",
    "Iringmara Tea Estate (Cachar): A lush plantation area near Dwarbund, where endless tea bushes stretch across the landscape, showcasing the traditional lifestyle of tea garden communities.",
    "Borojalengha Tea Estate (Cachar): A historic tea garden known for strong Assam CTC tea production and calm natural beauty across the eastern Cachar plains.",
    "West Jalinga Tea Estate (Cachar): A major tea estate featuring picturesque estates and green valleys, ideal for scenic drives and countryside exploration.",
    "Kailashpur Tea Estate (Cachar): A traditional plantation where visitors can witness the charm of Assam's tea culture amid peaceful rural surroundings.",
    "Dwarbund Tea Estate (Cachar): Located near the famous Chatla wetlands, this tea estate offers a unique blend of plantation scenery and natural bird-rich landscapes.",
    "Koombergram Tea Estate (Cachar): One of the well-known tea gardens of the region, surrounded by villages, green hill slopes and quiet plantation roads.",
    "Rosekandy Tea Estate (Cachar): A popular historic tea garden contributing to Barak Valley’s strong tea identity.",
    
    "Lallamookh Tea Estate (Hailakandi): A major tea garden in Hailakandi district offering sweeping views of endless tea fields and peaceful rural life.",
    "Bandookmara Tea Estate (Hailakandi): A scenic plantation belt surrounded by hill forests and tea bushes, representing the heart of the district’s tea economy.",
    "Aenakhal Tea Estate (Hailakandi): A classic tea estate showcasing the plantation heritage of southern Assam.",
    "Burni Braes Tea Estate (Hailakandi): One of the largest and most important tea estates in the Hailakandi region.",
    
    "Baithakhal Tea Estate (Karimganj): A famous tea estate in Karimganj district known for its picturesque landscapes and historic labour settlements.",
    "Bhubrighat Tea Estate (Karimganj): A lush plantation area near Patherkandi, home to traditional tea-growing communities.",
    "Hattikhira Tea Estate (Karimganj): A peaceful estate surrounded by green hills, producing classic Assam tea.",
    
    "Tea Tourism Tip: The best time to explore Barak Valley tea gardens is from October to February when the weather is pleasant and the plantations are at their scenic best.",
    
    # --- TRAVEL TIPS ---
    "Best Time to Visit: November to February for pleasant weather, festivals, and outdoor sightseeing.",
    "Monsoon Advisory: Trekking to hill temples like Bhuban Pahar becomes challenging during June–August.",
    "Local Food Must-Try: Shilaer Shondesh, local fish curries, sweets, and tea from valley plantations."
]


    silchar_subcategories = {
        "religious": [],
        "nature": [],
        "lakes": [],
        "historical": [],
        "education": [],
        "transport": [],
        "shopping": [],
        "healthcare": [],
        "city_life": [],
        "travel_tips": [],
        "cachar": [],
        "karimganj": [],
        "hailakandi": [],
    }

    current_district = "cachar"
    for entry in silchar_data:
        entry_lower = entry.lower()
        entry_title_lower = entry_lower.split(":", 1)[0]

        if "karimganj" in entry_lower and "district" in entry_lower:
            current_district = "karimganj"
            continue
        if "hailakandi" in entry_lower and "district" in entry_lower:
            current_district = "hailakandi"
            continue
        if "local travel tips" in entry_lower:
            current_district = "travel_tips"
            continue

        if current_district in {"cachar", "karimganj", "hailakandi"}:
            silchar_subcategories[current_district].append(entry)

        if (
            "temple" in entry_lower
            or "mandir" in entry_lower
            or "bari" in entry_lower
            or "ashram" in entry_lower
            or "mission" in entry_lower
            or "mukam" in entry_lower
            or "akhra" in entry_lower
            or "iskcon" in entry_lower
        ):
            silchar_subcategories["religious"].append(entry)

        # Only add to nature if the title contains nature-related keywords
        # but exclude if it's a temple or tunnel with a lake in description
        is_nature = False
        
        # Check for lake but exclude if it's a temple, tunnel, or park
        if "lake" in entry_title_lower:
            if ("temple" not in entry_title_lower and 
                "tunnel" not in entry_title_lower and
                "park" not in entry_title_lower):
                is_nature = True
                # Add to lakes category if it's specifically a lake
                silchar_subcategories["lakes"].append(entry)
        
        # Other nature keywords
        if not is_nature and (
            "park" in entry_title_lower
            or "hill" in entry_title_lower
            or "wetland" in entry_title_lower
            or "tea garden" in entry_title_lower
            or "garden" in entry_title_lower
            or "lake" in entry_title_lower
        ):
            is_nature = True
            
        if is_nature:
            silchar_subcategories["nature"].append(entry)

        if (
            "ruins" in entry_lower
            or "fort" in entry_lower
            or "capital" in entry_lower
            or "tunnel" in entry_lower
            or "gate" in entry_lower
            or "king" in entry_lower
            or "mughal" in entry_lower
            or "myth" in entry_lower
            or "mythological" in entry_lower
            or "histor" in entry_lower
        ):
            silchar_subcategories["historical"].append(entry)

        if (
            (
                "college" in entry_title_lower
                or "university" in entry_title_lower
                or "school" in entry_title_lower
                or "institute" in entry_title_lower
                or re.search(r"\bnit\b", entry_title_lower)
            )
            and "ashram" not in entry_lower
            and "vihar" not in entry_lower
        ):
            silchar_subcategories["education"].append(entry)

        if "airport" in entry_lower or "railway station" in entry_lower or "station" in entry_lower:
            silchar_subcategories["transport"].append(entry)

        # Shopping: match only on the title to avoid false positives from descriptions (e.g., temples near a market)
        if (
            "mall" in entry_title_lower
            or "market" in entry_title_lower
            or "bazar" in entry_title_lower
            or "bazaar" in entry_title_lower
        ):
            silchar_subcategories["shopping"].append(entry)

        if "hospital" in entry_lower or "medical" in entry_lower:
            silchar_subcategories["healthcare"].append(entry)

        if "club" in entry_lower or "library" in entry_lower or "ground" in entry_lower:
            silchar_subcategories["city_life"].append(entry)

        if current_district == "travel_tips" or "best time" in entry_lower or "local food" in entry_lower:
            silchar_subcategories["travel_tips"].append(entry)

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
    """
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
        ]
    )

    unclear_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Silchar Tourism Assistant. The user message is unclear or too short. "
                "Use the provided GENERAL_SILCHAR_INFO to give a brief helpful overview, then ask 1-2 clarifying questions "
                "to understand what they want (places, food, history, travel, shopping, hospitals, colleges, etc.). "
                "Keep it concise and tourist-friendly.",
            ),
            ("human", "GENERAL_SILCHAR_INFO:\n{general_info}\n\nUser message:\n{input}"),
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
    if user_input := st.chat_input("Ask about Silchar places OR type a category to list them (temples/religious, nature, historical, education, transport, shopping, healthcare, city_life, travel_tips) or a district (cachar, karimganj, hailakandi)."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            normalized_input = " ".join(user_input.strip().lower().split())
            first_word = normalized_input.split(" ", 1)[0] if normalized_input else ""
            tokens = [t for t in normalized_input.split(" ") if t]

            # Check for general queries about Silchar
            general_queries = [
                'everything about silchar',
                'all info about silchar',
                'tell me about silchar',
                'what can you tell me about silchar',
                'show me all about silchar',
                'all about silchar',
                'complete information about silchar',
                'full details about silchar',
                'silchar information',
                'silchar details',
                'silchar info',
                'silchar',
            ]

            if any(query in normalized_input for query in general_queries):
                response = f"{GENERAL_SILCHAR_INFO}\n\n"
                response += "Here's a comprehensive list of places and information about Silchar and its surroundings:\n\n"
                for i, entry in enumerate(silchar_data, 1):
                    title = entry.split(':', 1)[0].strip()
                    response += f"{i}. {title}\n"
                response += "\nThis is just a summary. Would you like more details about any specific place or aspect of Silchar? " \
                          "You can ask about categories like 'temples', 'lakes', 'shopping', 'education', etc., " \
                          "or ask about a specific place by name."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            category_aliases = {
                "temple": "religious",
                "temples": "religious",
                "religious": "religious",
                "nature": "nature",
                "parks": "nature",
                "park": "nature",
                "lakes": "nature",
                "lake": "nature",
                "hills": "nature",
                "hill": "nature",
                "history": "historical",
                "historical": "historical",
                "education": "education",
                "colleges": "education",
                "college": "education",
                "universities": "education",
                "university": "education",
                "transport": "transport",
                "travel": "travel_tips",
                "tips": "travel_tips",
                "food": "travel_tips",
                "shopping": "shopping",
                "market": "shopping",
                "markets": "shopping",
                "bazar": "shopping",
                "bazaar": "shopping",
                "bazars": "shopping",
                "bazaars": "shopping",
                "mall": "shopping",
                "malls": "shopping",
                "health": "healthcare",
                "hospital": "healthcare",
                "hospitals": "healthcare",
                "healthcare": "healthcare",
                "city": "city_life",
                "clubs": "city_life",
                "district": None,
                "cachar": "cachar",
                "karimganj": "karimganj",
                "hailakandi": "hailakandi",
            }

            requested_category = None
            if normalized_input in silchar_subcategories:
                requested_category = normalized_input
            elif first_word in category_aliases and category_aliases[first_word]:
                requested_category = category_aliases[first_word]
            else:
                for t in tokens:
                    if t in category_aliases and category_aliases[t]:
                        requested_category = category_aliases[t]
                        break

            if requested_category and requested_category in silchar_subcategories:
                items = silchar_subcategories[requested_category]
                if items:
                    answer = "\n".join([f"- {i}" for i in items])
                else:
                    answer = "No items found for that category yet."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()
            location_hint_tokens = {
                "lake",
                "temple",
                "mandir",
                "park",
                "college",
                "university",
                "airport",
                "station",
                "fort",
                "market",
                "river",
                "garden",
                "hill",
            }

            matches_knowledge_base = False
            if normalized_input:
                for entry in silchar_data:
                    entry_lower = entry.lower()
                    if normalized_input in entry_lower:
                        matches_knowledge_base = True
                        break
                    if len(tokens) == 1 and tokens[0] and tokens[0] in entry_lower:
                        matches_knowledge_base = True
                        break
            looks_like_place_query = (
                len(tokens) >= 2
                or any(t in location_hint_tokens for t in tokens)
                or matches_knowledge_base
                or "?" in normalized_input
            )
            is_unclear = (
                (len(normalized_input) < 10 and not looks_like_place_query)
                or first_word in {"hi", "hello", "hey", "hii", "hlo"}
                or "help" in normalized_input
                or "information" in normalized_input
                or normalized_input in {"silchar", "silchar?"}
                or normalized_input.endswith(" about silchar")
                or normalized_input.startswith("tell me about silchar")
                or normalized_input.startswith("about silchar")
            )

            if is_unclear:
                unclear_messages = unclear_prompt_template.format_messages(
                    input=user_input,
                    general_info=GENERAL_SILCHAR_INFO,
                )
                unclear_response = llm.invoke(unclear_messages)
                answer = getattr(unclear_response, "content", str(unclear_response))
            else:
                # The new rag_chain returns a dictionary with an "answer" key
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to begin your journey through Silchar!")