import os
import requests
from typing import List, Dict, Any, Optional
import PyPDF2
from io import BytesIO
import chromadb
import hashlib
import time
import threading

class StartupVectorChatAgent:
    def __init__(self):
        print("🚀 Initializing Startup Vector Chat Agent...")
        
        # Lazy initialization for heavy libraries
        self._llm = None
        self._agent = None
        
        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "hr_documents_startup"
        self.collection = None
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # Store functions for later use
        self._list_files_in_folder = None
        self._get_signed_url = None
        self._check_file_exists = None
        
        print("✅ Agent initialized - ready for preloading")
    
    def set_gcs_functions(self, list_files_in_folder, get_signed_url, check_file_exists):
        """Set GCS functions for use during initialization"""
        self._list_files_in_folder = list_files_in_folder
        self._get_signed_url = get_signed_url
        self._check_file_exists = check_file_exists
    
    def preload_vector_store_background(self):
        """Preload vector store in background thread"""
        def preload():
            try:
                if self._list_files_in_folder is None:
                    print("⚠️ GCS functions not set, skipping preload")
                    return
                
                print("🔄 Background: Starting vector store preload...")
                with self.initialization_lock:
                    success = self.initialize_vector_store_fast(
                        self._list_files_in_folder, 
                        self._get_signed_url, 
                        self._check_file_exists
                    )
                    if success:
                        print("✅ Background: Vector store preloaded successfully!")
                        self.is_initialized = True
                    else:
                        print("❌ Background: Vector store preload failed")
            except Exception as e:
                print(f"❌ Background preload error: {e}")
        
        # Start background thread
        thread = threading.Thread(target=preload, daemon=True)
        thread.start()
        return thread
    
    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Faster than GPT-4
                temperature=0.3,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        return self._llm
    
    @property
    def agent(self):
        if self._agent is None:
            from crewai import Agent
            self._agent = Agent(
                role="HR Assistant",
                goal="Provide quick, accurate answers",
                backstory="You are a helpful HR assistant.",
                verbose=False,
                allow_delegation=False,
                llm=self.llm
            )
        return self._agent
    
    def extract_text_from_pdf_fast(self, pdf_url: str) -> str:
        """Fast PDF extraction with limits"""
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            # Only first 3 pages for speed
            max_pages = min(3, len(pdf_reader.pages))
            for page in pdf_reader.pages[:max_pages]:
                page_text = page.extract_text()
                text += page_text + "\n"
                # Stop if we have enough content
                if len(text) > 2500:
                    break
            
            return text.strip()
        except Exception as e:
            print(f"❌ Error extracting PDF: {e}")
            return ""
    
    def chunk_text_simple(self, text: str, chunk_size: int = 600) -> List[str]:
        """Simple, fast text chunking"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50:  # Only add meaningful chunks
                chunks.append(chunk)
        
        return chunks
    
    def get_documents_hash(self, list_files_in_folder) -> str:
        """Generate hash to check if documents have changed"""
        all_files = []
        for folder in ["policies/", "benefits/"]:
            try:
                files = list_files_in_folder(folder)
                all_files.extend([f for f in files if f.endswith('.pdf')])
            except:
                pass
        
        hash_input = "|".join(sorted(all_files))
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def initialize_vector_store_fast(self, list_files_in_folder, get_signed_url, check_file_exists) -> bool:
        """Fast vector store initialization"""
        try:
            current_hash = self.get_documents_hash(list_files_in_folder)
            print("🔄 Checking existing vector store...")
            
            # Try to get existing collection
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                # Check if collection has data and hash matches
                metadata = self.collection.get(include=['metadatas'])
                if metadata['metadatas'] and len(metadata['metadatas']) > 0:
                    stored_hash = metadata['metadatas'][0].get('documents_hash')
                    if stored_hash == current_hash:
                        count = self.collection.count()
                        print(f"✅ Using existing vector store ({count} chunks)")
                        return True
                
                # Hash doesn't match, delete old collection
                self.chroma_client.delete_collection(self.collection_name)
                print("🔄 Documents changed, rebuilding...")
                
            except:
                print("📝 Creating new vector store...")
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load documents quickly
            documents = self.load_documents_parallel(list_files_in_folder, get_signed_url, check_file_exists)
            
            if not documents:
                print("❌ No documents found")
                return False
            
            # Process into chunks
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            chunk_id = 0
            for doc_name, content in documents.items():
                chunks = self.chunk_text_simple(content)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source": doc_name,
                        "chunk_index": i,
                        "doc_type": "policy" if "policy" in doc_name.lower() else "benefit",
                        "documents_hash": current_hash if chunk_id == 0 else ""  # Store hash only once
                    })
                    all_ids.append(f"{doc_name}_chunk_{i}")
                    chunk_id += 1
            
            # Add all chunks at once
            if all_chunks:
                self.collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                
                print(f"✅ Vector store created: {len(all_chunks)} chunks from {len(documents)} docs")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Vector store initialization error: {e}")
            return False
    
    def load_documents_parallel(self, list_files_in_folder, get_signed_url, check_file_exists) -> Dict[str, str]:
        """Load documents with some optimization"""
        documents = {}
        
        for folder in ["policies/", "benefits/"]:
            try:
                files = list_files_in_folder(folder)
                print(f"📁 Processing {len(files)} files from {folder}")
                
                for file_path in files:
                    if file_path.endswith('.pdf') and check_file_exists(file_path):
                        signed_url = get_signed_url(file_path)
                        if signed_url:
                            content = self.extract_text_from_pdf_fast(signed_url)
                            if content:
                                file_name = file_path.split('/')[-1].replace('.pdf', '')
                                documents[file_name] = content
                                print(f"✅ Loaded: {file_name}")
                        
                        # Small delay to be gentle on GCS
                        time.sleep(0.1)
                        
            except Exception as e:
                print(f"❌ Error loading from {folder}: {e}")
        
        return documents
    
    def search_relevant_content_fast(self, question: str) -> List[Dict]:
        """Fast search with minimal results"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=2,  # Only top 2 results for speed
                include=['documents', 'metadatas']
            )
            
            relevant_content = []
            for i, doc in enumerate(results['documents'][0]):
                relevant_content.append({
                    'content': doc,
                    'source': results['metadatas'][0][i]['source']
                })
            
            return relevant_content
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def process_chat_request(self, question: str, list_files_in_folder, get_signed_url, check_file_exists) -> str:
        """Fast chat processing"""
        start_time = time.time()
        
        try:
            # Check if initialized, if not try quick initialization
            if not self.is_initialized:
                with self.initialization_lock:
                    if not self.is_initialized:
                        print("⚡ Quick vector store check...")
                        self.set_gcs_functions(list_files_in_folder, get_signed_url, check_file_exists)
                        if not self.initialize_vector_store_fast(list_files_in_folder, get_signed_url, check_file_exists):
                            return "I'm sorry, I'm still initializing. Please try again in a moment."
                        self.is_initialized = True
            
            # Fast search
            relevant_content = self.search_relevant_content_fast(question)
            
            if not relevant_content:
                return "I'm sorry, I couldn't find relevant information. Please contact HR directly."
            
            # Create minimal context
            context = ""
            for item in relevant_content:
                # Limit context size for speed
                content_preview = item['content'][:800] + "..." if len(item['content']) > 800 else item['content']
                context += f"From {item['source']}: {content_preview}\n\n"
                print(f"🎯 Using: {item['source']}")
            
            # Simple direct LLM call instead of CrewAI for speed
            prompt = f"""Based on this company information, answer the employee question concisely:

Question: {question}

Company Info:
{context}

Answer in 2-3 sentences, be helpful and reference the source document."""

            response = self.llm.invoke(prompt)
            
            total_time = time.time() - start_time
            print(f"⚡ Response time: {total_time:.1f}s")
            
            return response.content
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm experiencing technical difficulties. Please try again later."
