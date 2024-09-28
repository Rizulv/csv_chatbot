import os
import pickle
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Function to fetch article details from the URL
def fetch_article_details(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to fetch article at {url}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No Title'
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        
        return {"title": title, "url": url, "content": content}
    
    except Exception as e:
        st.error(f"Error fetching article from {url}: {e}")
        return None

# Function to save article details into a CSV
def save_article_to_csv(article):
    df = pd.DataFrame([article])
    csv_filename = "temp_article.csv"
    df.to_csv(csv_filename, index=False)
    st.write(f"Article saved to {csv_filename}")
    return csv_filename

# Function to save embeddings to a pickle file
def store_csv_embeddings(csv_file, filename="csv_embeddings"):
    csv_file.seek(0)  # Reset file pointer to the beginning
    df = pd.read_csv(csv_file)

    if 'content' not in df.columns:
        st.error("The CSV file does not contain a 'content' column.")
        return None

    texts = df['content'].tolist()

    # Convert content into Document objects
    documents = [Document(page_content=text) for text in texts]

    # Create HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Generate and store embeddings in FAISS
    vectors = FAISS.from_documents(documents, embeddings)

    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)
    return vectors

# Function to retrieve or generate embeddings
def get_csv_embeddings(csv_file, filename="csv_embeddings"):
    if not os.path.isfile(filename + ".pkl"):
        vectors = store_csv_embeddings(csv_file, filename)
    else:
        with open(filename + ".pkl", "rb") as f:
            vectors = pickle.load(f)
    
    return vectors

# Conversational chat function
def conversational_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Setup the conversational chain
def setup_chain(csv_file, model='llama2'):
    vectors = get_csv_embeddings(csv_file)
    
    if vectors is None:
        return None
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model=model),
        retriever=vectors.as_retriever(),
        chain_type="stuff"
    )
    return chain

# Streamlit app UI
st.title("📄 Article to CSV to Chatbot")

# Step 1: User enters article link
st.header("Step 1: Enter an article link")
article_link = st.text_input("Enter the article link")

if st.button("Fetch and Generate CSV"):
    if article_link:
        # Fetch article details
        article_details = fetch_article_details(article_link)
        if article_details:
            # Save article to CSV
            csv_filename = save_article_to_csv(article_details)
            st.success(f"CSV file generated: {csv_filename}")
            
            # Provide CSV download option
            with open(csv_filename, "rb") as file:
                st.download_button(label="Download CSV", data=file, file_name=csv_filename)
    else:
        st.error("Please enter a valid article link.")

# Step 2: Upload generated CSV
st.header("Step 2: Upload the generated CSV for chat")
uploaded_csv = st.file_uploader("Upload the CSV file you just downloaded", type="csv")

if uploaded_csv:
    try:
        # Reset file pointer and preview CSV content
        uploaded_csv.seek(0)
        df = pd.read_csv(uploaded_csv)
        st.write("CSV Preview:")
        st.write(df.head())  # Show the first few rows of the CSV

        # Step 3: Setup chain and embeddings
        chain = setup_chain(uploaded_csv)

        if chain is not None:
            st.success("Chatbot is ready! Ask your question below.")
            
            # History to store conversation
            conversation_history = []
            
            # User input for question
            user_query = st.text_input("Ask a question about the CSV content")
            
            if user_query:
                # Generate a conversational response
                response = conversational_chat(user_query, chain, conversation_history)
                st.write(f"Bot: {response}")
    except Exception as e:
        st.error(f"Error processing the CSV: {e}")
