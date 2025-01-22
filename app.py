#writefile app.py
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
memory = MemorySaver()

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2nRjklHRNgND98qIvCV7KNsmHLBI9yA4"
genai.configure(api_key="AIzaSyC2nRjklHRNgND98qIvCV7KNsmHLBI9yA4")
llm2 = genai.GenerativeModel('models/gemini-1.5-flash')
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash")

# Step 1: Load and encode the FAQ data using SentenceTransformer
model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

combined_data = {
    "faqs": [
        {"question": "What is the return policy?", "answer": "To return a product, log in to your account, select 'Orders', and click on 'Return'. Items must be returned within 30 days of delivery in original packaging. Once received, refund will be processed within 5-7 business days."},
        {"question": "How do I track my order?", "answer": "To track your order, go to 'My Orders' and select 'Track Order'. You'll receive a tracking number via email once your order ships. You can also enable push notifications for real-time updates."},
        {"question": "What payment methods do you accept?", "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, Google Pay, and bank transfers. All transactions are secured with 256-bit encryption."},
        {"question": "How long does shipping take?", "answer": "Standard shipping takes 3-5 business days. Express shipping delivers within 1-2 business days. International shipping may take 7-14 business days. Free shipping is available on orders over $50."},
        {"question": "Do you ship internationally?", "answer": "Yes, we ship to over 100 countries. International shipping costs vary by location and package weight. Customs fees and import duties may apply and are the responsibility of the customer."},
        {"question": "How do I cancel my order?", "answer": "To cancel an order, go to 'My Orders', select the order you want to cancel, and click 'Cancel Order'. Orders can only be cancelled within 1 hour of placement or before shipping, whichever comes first."},
        {"question": "What is your size guide?", "answer": "Our size guide can be found on each product page under 'Size Guide'. It includes detailed measurements for all sizes. For the best fit, measure yourself and compare with our size chart."},
        {"question": "Do you offer gift wrapping?", "answer": "Yes, gift wrapping is available for $5 per item. You can select this option during checkout. We also provide gift receipts and the ability to add personalized messages."},
        {"question": "How can I change my delivery address?", "answer": "To change your delivery address, go to 'My Orders', select the order, and click 'Update Delivery Address'. This is only possible before the order has been shipped."},
        {"question": "What is your price matching policy?", "answer": "We offer price matching on identical items from major authorized retailers. Submit a price match request within 14 days of purchase with proof of the lower price."},
        #{"question": "Do you have a loyalty program?", "answer": "Yes, our loyalty program rewards you with 1 point for every $1 spent. Points can be redeemed for discounts, free shipping, and exclusive offers. Sign up is free through your account."},
        {"question": "How do I apply a coupon code?", "answer": "Enter your coupon code in the 'Promo Code' box during checkout before clicking 'Place Order'. Only one coupon code can be used per order unless otherwise specified."},
        {"question": "What is your warranty policy?", "answer": "Most products come with a standard 1-year warranty against manufacturing defects. Extended warranties are available for purchase on select items. Check individual product pages for specific warranty information."},
        {"question": "i want to track my order and want to know where did the product reached", "answer": "your product have reached ahemdabad, and will reach to mumbai in 6 hrs. please be ready to recive the order. for further contact reach us out at amazon.com"},
        {"question": "order tracking #123?", "answer": "The order is dispatched and you will get in 5 days"},
        {"question": "i have order no #154 get my order details", "answer": "you have ordered a boat headphone with dual connection(bluetootth and wifi) color of the product is red, weights is 100grams, order owner is akash, and payment way is cash on delivery"},
        {"question": "Smart Home Security Camera", "answer": "Smart Home Security Camera is 1080p HD video, night vision, motion detection, two-way audio, and cloud storage. Compatible with Alexa and Google Assistant."},
        {"question": "Wireless Bluetooth Headphones", "answer": "Wireless Bluetooth Headphones is Up to 20 hours of playtime, quick charging, sweat-resistant, and ergonomic design."},
        {"question": "Men's Running Shoes", "answer":"Men's Running Shoes Lightweight, breathable, and durable. Suitable for road and trail running. Available in sizes 7-13."},
        {"question": "Organic Cotton T-Shirt", "answer":"Organic Cotton T-Shirt 100% certified organic cotton, soft and breathable. Available in multiple colors and sizes."},
        {"question": "Fitness Equipment Weight", "answer":"Fitness Equipment Weight range: 5-50 lbs per dumbbell. Quick-adjust dial system for easy weight changes"},
        {"question": "Yoga Mat", "answer":"Yoga Mat Eco-friendly, non-slip, and extra thick for comfort. Available in multiple colors."},
        {"question": "Leather Wallet", "answer":"Leather Wallet Genuine leather, slim design with multiple card slots and a cash compartment."},
        {"question": "Stainless Steel Water Bottle", "answer":"Stainless Steel Water Bottle 24-ounce capacity, keeps beverages cold for 24 hours and hot for 12 hours. Dishwasher safe"},
    ],
    "products": [
        {
            "category": "Electronics",
            "products": [
                {
                    "name": "Smart Home Security Camera",
                    "description": "Smart Home Security Camera is 1080p HD video, night vision, motion detection, two-way audio, and cloud storage. Compatible with Alexa and Google Assistant.",
                    "price": 89.99,
                    "stock": 150,
                    "sku": "ELEC001",
                    "faqs": [
                        {"question": "Smart Home Security Camera is Can I use the Smart Home Security Camera outdoors?", "answer": "No, it is designed for indoor use only. For outdoor security, we recommend our Outdoor Security Camera."},
                        {"question": "Smart Home Security Camera is Does it require a subscription for cloud storage?", "answer": "Basic cloud storage is free for 7 days. Extended plans are available for purchase."}
                    ]
                },
                {
                    "name": "Wireless Bluetooth Headphones",
                    "description": "Wireless Bluetooth Headphones is Up to 20 hours of playtime, quick charging, sweat-resistant, and ergonomic design.",
                    "price": 59.99,
                    "stock": 200,
                    "sku": "ELEC002",
                    "faqs": [
                        {"question": "Wireless Bluetooth Headphones is Are the headphones sweat-resistant?", "answer": "Yes, they are sweat-resistant, making them ideal for workouts."},
                        {"question": "Wireless Bluetooth Headphones is What is the range of the Bluetooth connection?", "answer": "The range is up to 33 feet (10 meters) in open space."}
                    ]
                }
            ]
        },
        {
            "category": "Clothing",
            "products": [
                {
                    "name": "Men's Running Shoes",
                    "description": "Men's Running Shoes Lightweight, breathable, and durable. Suitable for road and trail running. Available in sizes 7-13.",
                    "price": 79.99,
                    "stock": 120,
                    "sku": "CLOTH001",
                    "faqs": [
                        {"question": "Mens running shoes Are these shoes suitable for trail running?", "answer": "Yes, they are designed for both road and trail running."},
                        {"question": "Mens running shoes Do they come in wide sizes?", "answer": "Yes, wide sizes are available for select sizes."}
                    ]
                },
                {
                    "name": "Organic Cotton T-Shirt",
                    "description": "Organic Cotton T-Shirt 100% certified organic cotton, soft and breathable. Available in multiple colors and sizes.",
                    "price": 29.99,
                    "stock": 250,
                    "sku": "CLOTH002",
                    "faqs": [
                        {"question": "Organic cotton t-shirt Is the T-shirt machine washable?", "answer": "Yes, it is machine washable. We recommend cold water and air drying."},
                        {"question": "Organic cotton t-shirt Does it shrink after washing?", "answer": "No, it is pre-shrunk to maintain its size and shape."}
                    ]
                }
            ]
        },
        {
            "category": "Fitness Equipment",
            "products": [
                {
                    "name": "Adjustable Dumbbell Set",
                    "description": "Fitness Equipment Weight range: 5-50 lbs per dumbbell. Quick-adjust dial system for easy weight changes.",
                    "price": 199.99,
                    "stock": 60,
                    "sku": "FIT001",
                    "faqs": [
                        {"question": "Adjustable dumbbel set hat is the weight range?", "answer": "Each dumbbell can be adjusted from 5 to 50 pounds."},
                        {"question": "Adjustable dumbbel Is it easy to adjust the weights?", "answer": "Yes, it features a quick-adjust dial system for easy changes."}
                    ]
                },
                {
                    "name": "Yoga Mat",
                    "description": "Yoga Mat Eco-friendly, non-slip, and extra thick for comfort. Available in multiple colors.",
                    "price": 39.99,
                    "stock": 300,
                    "sku": "FIT002",
                    "faqs": [
                        {"question": "Yoga mat Is the mat eco-friendly?", "answer": "Yes, it is made from eco-friendly materials."},
                        {"question": "Yoga mat What is the thickness of the mat?", "answer": "The mat is 6mm thick for added comfort."}
                    ]
                }
            ]
        },
        {
            "category": "Accessories",
            "products": [
                {
                    "name": "Stainless Steel Water Bottle",
                    "description": "Stainless Steel Water Bottle24-ounce capacity, keeps beverages cold for 24 hours and hot for 12 hours. Dishwasher safe.",
                    "price": 19.99,
                    "stock": 500,
                    "sku": "ACC001",
                    "faqs": [
                        {"question": "Stainless steel water bottle Is the water bottle dishwasher safe?", "answer": "Yes, it is dishwasher safe."},
                        {"question": "Stainless steel water bottle What is the capacity?", "answer": "The capacity is 24 ounces (710 ml)."}
                    ]
                },
                {
                    "name": "Leather Wallet",
                    "description": "Leather Wallet Genuine leather, slim design with multiple card slots and a cash compartment.",
                    "price": 34.99,
                    "stock": 200,
                    "sku": "ACC002",
                    "faqs": [
                        {"question": "Leather wallet Is the wallet made of genuine leather?", "answer": "Yes, it is made of 100% genuine leather."},
                        {"question": "Leather wallet How many card slots does it have?", "answer": "It has 6 card slots and a cash compartment."}
                    ]
                }
            ]
        }
    ]
}

# Extract all FAQ questions, answers, and product details
faq_data = []

# Add general FAQs
for faq in combined_data["faqs"]:
    faq_data.append({
        "question": faq["question"],
        "answer": faq["answer"],
        "type": "general",  # Indicates this is a general FAQ
        "product": None  # No associated product
    })

# Add product-specific FAQs
for category in combined_data["products"]:
    for product in category["products"]:
        for faq in product.get("faqs", []):
            faq_data.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "type": "product",  # Indicates this is a product-specific FAQ
                "product": {  # Include product details
                    "name": product["name"],
                    "description": product["description"],
                    "price": product["price"],
                    "stock": product["stock"],
                    "sku": product["sku"],
                    "category": category["category"]
                }
            })

# Encode questions, answers, and product details into embeddings
faq_texts = [
    f"Question: {item['question']} Answer: {item['answer']} Product: {item['product']['name'] if item['product'] else 'N/A'}"
    for item in faq_data
]
faq_embeddings = np.array(model.embed_documents(faq_texts))

# Build FAISS index for FAQ retrieval
dimension = faq_embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(faq_embeddings)  # Add embeddings to the index

# Save the index for later use (optional)
faiss.write_index(index, "faq_index.faiss")

# Function to retrieve relevant FAQ based on a query
def retrieve_faq(query, top_k=2):
    #query_embedding = model.encode([query])
    #distances, indices = index.search(query_embedding, top_k)
    #relevant_faqs = [faq_data[i] for i in indices[0]]
    #return relevant_faqs

    query_embedding = np.array(model.embed_query(query))
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    # Retrieve the most relevant FAQs with full context
    relevant_faqs = [faq_data[i] for i in indices[0]]
    #revelant_faqs = str(revelant_faqs[0]) + " " + str(revelant_faq[1])
    return relevant_faqs

# Step 2: Define tools for intent classification, NER, and FAQ handling
# Intent Classification Tool
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Classify the intent of the following sentence into one of these categories: "
        "1. Product search, 2. FAQ inquiry, 3. Order tracking, 4. General chat. "
        "The sentence is: {user_input}"
    ),
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

def classify_intent(user_input):
    response = intent_chain.run(user_input)
    # Extract the intent from the response (e.g., "Product search")
    if "Product search" in response:
        return "Product search"
    elif "FAQ inquiry" in response:
        return "FAQ inquiry"
    elif "Order tracking" in response:
        return "Order tracking"
    elif "General chat" in response:
        return "General chat"
    else:
        return "General chat"  # Default to general chat

# Named Entity Recognition (NER) Tool
ner_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Perform named entity recognition on the following e-commerce sentence: {user_input}. "
        "Extract entities like product names, locations, and stores."
    ),
)
ner_chain = LLMChain(llm=llm, prompt=ner_prompt)

def perform_ner(user_input):
    return ner_chain.run(user_input)

# FAQ Inquiry Tool (RAG tool)
faq_prompt = PromptTemplate(
    input_variables=["retrieved_faq", "user_input"],
    template=(
        "Here is an FAQ relevant to the user's query: {retrieved_faq}. "
        "Explain this FAQ in more detail to the customer based on their input: {user_input}."
    ),
)
faq_chain = LLMChain(llm=llm, prompt=faq_prompt)

def handle_faq(user_input, retrieved_faq):
    return faq_chain.run({"user_input": user_input, "retrieved_faq": retrieved_faq})

# Define the state
class State(TypedDict):
    #user_input: str
    intent: str
    ner_result: str
    retrieved_faq: str
    response: str
    messages: Annotated[list, add_messages]
    #memory: list  # Add memory to store conversation history

# Add nodes
def intent_classification(state: State) -> State:
    intent = classify_intent(state["messages"][-1])
    return {"intent": intent}

def ner_product_search(state: State) -> State:
    ner_result = perform_ner(state["messages"][-1])
    return {"ner_result": ner_result}

def product_search(state: State) -> State:
    retrieved_faq = retrieve_faq(str(state["messages"][-1]))
    retrieved_faq = str(retrieved_faq[0]) + " " + str(retrieved_faq[1])
    return {"retrieved_faq": retrieved_faq}

def llm_product_search(state: State) -> State:
    response = handle_faq(state["messages"][-1], state["retrieved_faq"][-1])
    return {"response": response}

def general_chat(state: State) -> State:
    #print(state['messages'])
    string = " "
    for i in range(len(state["messages"])-1):
      string += str(state["messages"][i])

    string += "Only this part is the current question being asked before other than this statement everything is Previous chat history hence only ansswer this question:"
    string += str(state["messages"][-1])
    response = llm2.generate_content(string).text

    return {"response": response}

def faq_inquiry(state: State) -> State:
    retrieved_faq = retrieve_faq(str(state["messages"][-1]))
    retrieved_faq = str(retrieved_faq[0]) + " " + str(retrieved_faq[1])
    return {"retrieved_faq": retrieved_faq}

def faq_answer(state: State) -> State:
    response = handle_faq(state["messages"][-1], state["retrieved_faq"][-1])
    return {"response": response}

def ner_order_tracking(state: State) -> State:
    ner_result = perform_ner(state["messages"][-1])
    return {"ner_result": ner_result}

def order_search(state: State) -> State:
    retrieved_faq = retrieve_faq(str(state["ner_result"][-1]))
    retrieved_faq = str(retrieved_faq[0]) + " " + str(retrieved_faq[1])
    return {"retrieved_faq": retrieved_faq}

def llm_order_tracking(state: State) -> State:
    response = handle_faq(state["messages"][-1], state["retrieved_faq"][-1])
    return {"response": response}

#def end(state: State) -> State:
    # Add the final response to memory
    #state["memory"].append({"role": "assistant", "content": state["response"]})
    #return {"response": "Goodbye!"} #, "memory": state["memory"]}

# Initialize the graph
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("INTENT_CLASSIFICATION", intent_classification)
graph.add_node("NER_PRODUCT_SEARCH", ner_product_search)
graph.add_node("PRODUCT_SEARCH", product_search)
graph.add_node("LLM_PRODUCT_SEARCH", llm_product_search)
graph.add_node("GENERAL_CHAT", general_chat)
graph.add_node("FAQ_INQUIRY", faq_inquiry)
graph.add_node("FAQ_ANSWER", faq_answer)
graph.add_node("NER_ORDER_TRACKING", ner_order_tracking)
graph.add_node("ORDER_SEARCH", order_search)
graph.add_node("LLM_ORDER_TRACKING", llm_order_tracking)
#graph.add_node("END", end)

# Add edges
graph.add_conditional_edges(
    source="INTENT_CLASSIFICATION",
    path=lambda state: state["intent"],
    path_map={
        "Product search": "NER_PRODUCT_SEARCH",
        "General chat": "GENERAL_CHAT",
        "FAQ inquiry": "FAQ_INQUIRY",
        "Order tracking": "NER_ORDER_TRACKING"
    }
)

graph.add_edge("NER_PRODUCT_SEARCH", "PRODUCT_SEARCH")
graph.add_edge("PRODUCT_SEARCH", "LLM_PRODUCT_SEARCH")
#graph.add_edge("LLM_PRODUCT_SEARCH", "END")

#graph.add_edge("GENERAL_CHAT", "END")

graph.add_edge("FAQ_INQUIRY", "FAQ_ANSWER")
#graph.add_edge("FAQ_ANSWER", "END")

graph.add_edge("NER_ORDER_TRACKING", "ORDER_SEARCH")
graph.add_edge("ORDER_SEARCH", "LLM_ORDER_TRACKING")
#graph.add_edge("LLM_ORDER_TRACKING", "END")

# Set the entry point
graph.set_entry_point("INTENT_CLASSIFICATION")

# Compile the graph
graph = graph.compile(checkpointer = memory)

def chatbot_agent():
    config = {"configurable": {"thread_id": "1"}}
    st.session_state.messages = st.session_state.get("messages", [])

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("You: "):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create expandable section for node outputs
        with st.expander("View Agent Processing Steps"):
            # Prepare the progress placeholder
            progress_placeholder = st.empty()
            
            # Stream the events and process each node's output
            events = graph.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config,
                stream_mode="values",
            )
            
            final = {}
            for event in events:
                # Display intent classification result
                
                st.write("Processing...")
                final = event

            if "intent" in event:
                st.write("🎯 Intent Classification:", final["intent"])
                
            # Display NER results
            if "ner_result" in event:
                st.write("🔍 Named Entity Recognition:", final["ner_result"])
                
            # Display FAQ retrieval
            if "retrieved_faq" in event:
                st.write("📚 Retrieved FAQ:", final["retrieved_faq"])
                
            # Display final responses
            if "response" in event:
                st.write("💬 Response Generated:", final["response"])
                #Add response to chat
                st.session_state.messages.append({"role": "assistant", "content": final["response"]})
                with st.chat_message("assistant"):
                    st.markdown(event["response"])
                
            # Display message events
            if "messages" in event:
                message = event["messages"][-1]
                if isinstance(message, dict) and "content" in message:
                    st.write("📨 Message:", message["content"])
                    # Add message to chat
                    st.session_state.messages.append({"role": "assistant", "content": message["content"]})
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

# Streamlit UI setup
def main():
    st.title("E-commerce Chatbot Interface")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = "Hello! I'm your e-commerce assistant. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Add sidebar for debugging info (optional)
    with st.sidebar:
        st.subheader("Debug Information")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Run the chatbot agent
    chatbot_agent()

if __name__ == "__main__":
    main()
