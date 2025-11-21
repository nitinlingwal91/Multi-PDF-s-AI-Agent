import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
FAISS_INDEX_PATH = "faiss_index"
DEFAULT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.3
SIMILARITY_K = 3

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = False
    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False
    if 'pdf_names' not in st.session_state:
        st.session_state.pdf_names = []

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents.
    
    Args:
        pdf_docs: List of uploaded PDF files
        
    Returns:
        str: Concatenated text from all PDFs
    """
    pdf_text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text
        return pdf_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to split text into chunks
def get_text_chunks(text):
    """
    Split text into smaller chunks for processing.
    
    Args:
        text: Input text string
        
    Returns:
        list: List of text chunks
    """
    if not text or text.strip() == "":
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    """
    Create and save FAISS vector store from text chunks.
    
    Args:
        text_chunks: List of text chunks
        
    Returns:
        bool: Success status
    """
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

# Function to create conversational chain
def get_conversational_chain():
    """
    Create a conversational QA chain with custom prompt.
    
    Returns:
        Chain: LangChain QA chain
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided
    context, just say "I cannot find the answer in the provided documents",
    don't provide incorrect information.
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    try:
        model = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

# Function to handle user input and generate response
def user_input(user_question):
    """
    Process user question and generate answer from vector store.
    
    Args:
        user_question: User's question string
    """
    if not user_question or user_question.strip() == "":
        st.warning("Please enter a valid question.")
        return
    
    try:
        # Load embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Check if vector store exists
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error("Please upload and process PDF files first.")
            return
        
        # Load vector store with allow_dangerous_deserialization
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Perform similarity search
        docs = vector_store.similarity_search(user_question, k=SIMILARITY_K)
        
        if not docs:
            st.warning("No relevant information found in the documents.")
            return
        
        # Get conversational chain and generate response
        chain = get_conversational_chain()
        if chain is None:
            return
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Display response
        st.write("### 💬 Answer:")
        st.write(response["output_text"])
        
        # Store in conversation history
        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": response["output_text"]
        })
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# Function to validate API key
def validate_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
        st.info("Create a .env file in your project directory with: OPENAI_API_KEY=your-api-key-here")
        return False
    return True

# Main function
def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Multi-PDF Chat Agent",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Validate API key
    if not validate_api_key():
        st.stop()
    
    # Header
    st.title("📚 Multi-PDF Chat Agent 🤖")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Sidebar for PDF upload
    with st.sidebar:
        # Display robot image if it exists
        if os.path.exists("img/Robot.jpg"):
            st.image("img/Robot.jpg")
        
        st.markdown("---")
        st.title("📁 PDF Upload Section")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if pdf_docs:
            st.info(f"📄 {len(pdf_docs)} file(s) selected")
            
        if st.button("🔄 Process PDFs", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs... This may take a moment."):
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs.")
                        return
                    
                    # Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("Could not split text into chunks.")
                        return
                    
                    st.info(f"Created {len(text_chunks)} text chunks")
                    
                    # Create vector store
                    success = get_vector_store(text_chunks)
                    
                    if success:
                        st.session_state.processed_pdfs = True
                        st.session_state.vector_store_ready = True
                        st.session_state.pdf_names = [pdf.name for pdf in pdf_docs]
                        st.success("✅ PDFs processed successfully!")
                    else:
                        st.error("Failed to process PDFs.")
        
        # Display processed PDFs
        if st.session_state.processed_pdfs and st.session_state.pdf_names:
            st.markdown("---")
            st.subheader("📋 Processed Files:")
            for idx, name in enumerate(st.session_state.pdf_names, 1):
                st.text(f"{idx}. {name}")
        
        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.processed_pdfs = False
            st.session_state.vector_store_ready = False
            st.session_state.pdf_names = []
            
            # Remove FAISS index if it exists
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
            
            st.success("All data cleared!")
            st.rerun()
    
    # Main chat interface
    if st.session_state.vector_store_ready:
        st.success("✅ PDFs loaded and ready for questions!")
    else:
        st.info("👈 Please upload and process PDF files to get started.")
    
    # Question input
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic discussed in the documents?",
        key="user_question_input"
    )
    
    # Process question
    if st.button("🔍 Get Answer", use_container_width=False) or user_question:
        if not st.session_state.vector_store_ready:
            st.warning("Please upload and process PDF files first.")
        elif user_question:
            user_input(user_question)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("📜 Conversation History")
        
        for idx, conv in enumerate(reversed(st.session_state.conversation_history), 1):
            with st.expander(f"Q{len(st.session_state.conversation_history) - idx + 1}: {conv['question'][:50]}..."):
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Answer:** {conv['answer']}")

if __name__ == "__main__":
    main()
