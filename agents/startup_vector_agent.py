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
        self._check_file_exists = None        # Track user sessions for personalized greetings
        self.user_sessions = {}
        
        # URL cache for fast document reference generation
        self.url_cache = {}  # {file_path: {'url': signed_url, 'expires_at': timestamp}}
        self.url_cache_duration = 3000  # 50 minutes (safe buffer from 1 hour expiration)
        
        # HR contact information for fallback scenarios
        self.hr_contact_email = "hr.support@company.com"
        self.hr_phone = "+1 (555) 123-4567"
        
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
    
    def get_cached_signed_url(self, file_path: str, get_signed_url) -> str:
        """Lightning-fast URL generation with intelligent caching"""
        current_time = time.time()
        
        # Check cache first (microsecond lookup)
        if file_path in self.url_cache:
            cache_entry = self.url_cache[file_path]
            if current_time < cache_entry['expires_at']:
                return cache_entry['url']  # Instant return from cache
        
        # Generate new URL only if needed
        try:
            new_url = get_signed_url(file_path, expiration=3600)  # 1 hour
            if new_url:
                self.url_cache[file_path] = {
                    'url': new_url,
                    'expires_at': current_time + self.url_cache_duration
                }
                return new_url
        except Exception as e:
            print(f"URL generation error for {file_path}: {e}")
        
        return ""
    
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
        # Updated folder structure with parent "onboarding agent/" folder
        for folder in ["onboarding agent/benefits/", "onboarding agent/expense/", "onboarding agent/onboarding/", "onboarding agent/policy/"]:
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
            for doc_name, (content, file_path) in documents.items():  # Unpack content and file path
                chunks = self.chunk_text_simple(content)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    
                    # Improved document type classification based on folder path
                    doc_type = "document"  # default
                    if "onboarding agent/policy/" in file_path.lower():
                        doc_type = "policy"
                    elif "onboarding agent/benefit" in file_path.lower():
                        doc_type = "benefit"
                    elif "onboarding agent/expense/" in file_path.lower():
                        doc_type = "expense"
                    elif "onboarding agent/onboarding/" in file_path.lower():
                        doc_type = "onboarding"
                    
                    all_metadatas.append({
                        "source": doc_name,
                        "file_path": file_path,  # Store the original GCS file path
                        "chunk_index": i,
                        "doc_type": doc_type,
                        "documents_hash": current_hash if chunk_id == 0 else ""  # Store hash only once
                    })
                    all_ids.append(f"{doc_name}_chunk_{i}")
                    chunk_id += 1
            
            # Add all chunks at once
            if all_chunks:
                self.collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids                )
                
                print(f"✅ Vector store created: {len(all_chunks)} chunks from {len(documents)} docs")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Vector store initialization error: {e}")
            return False
    
    def load_documents_parallel(self, list_files_in_folder, get_signed_url, check_file_exists) -> Dict[str, tuple]:
        """Load documents and store content with file paths"""
        documents = {}
        
        # Updated folder structure with parent "onboarding agent/" folder
        for folder in ["onboarding agent/benefits/", "onboarding agent/expense/", "onboarding agent/onboarding/", "onboarding agent/policy/"]:
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
                                documents[file_name] = (content, file_path)  # Store content AND file path
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
                n_results=3,  # Increased from 2 to 3 for better coverage
                include=['documents', 'metadatas']
            )
            
            relevant_content = []
            for i, doc in enumerate(results['documents'][0]):
                relevant_content.append({
                    'content': doc,
                    'source': results['metadatas'][0][i]['source'],
                    'file_path': results['metadatas'][0][i].get('file_path')  # Include file path
                })
            
            return relevant_content
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def process_chat_request(self, question: str, email: str, list_files_in_folder, get_signed_url, check_file_exists, supabase_client=None) -> str:
        """Fast chat processing with personalized greeting"""
        start_time = time.time()
        
        try:
            # Check if this is the user's first message for personalized greeting
            is_first_message = email not in self.user_sessions
            
            if is_first_message and supabase_client:
                # Get user profile for personalized greeting
                user_profile = self.get_user_profile(email, supabase_client)
                welcome_message = self.generate_welcome_message(user_profile)
                self.user_sessions[email] = True
                
                # If it's a simple greeting, return welcome message
                greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
                if any(word in question.lower() for word in greeting_words) and len(question.split()) <= 3:
                    return welcome_message
              # Check if initialized, if not try quick initialization
            if not self.is_initialized:
                with self.initialization_lock:
                    if not self.is_initialized:
                        print("⚡ Quick vector store check...")
                        self.set_gcs_functions(list_files_in_folder, get_signed_url, check_file_exists)
                        if not self.initialize_vector_store_fast(list_files_in_folder, get_signed_url, check_file_exists):
                            return "I'm currently initializing my knowledge base. Please try again in a moment, and I'll be ready to help you with your HR questions!"
                        self.is_initialized = True
              # Fast search
            relevant_content = self.search_relevant_content_fast(question)
            
            if not relevant_content:
                # Professional fallback with HR contact info
                fallback_message = f"""I apologize, but I couldn't find specific information related to your question in our current HR knowledge base.

**For personalized assistance, please contact our HR team:**
📧 Email: {self.hr_contact_email}
📞 Phone: {self.hr_phone}
🕒 Business Hours: Monday-Friday, 9:00 AM - 5:00 PM EST

I can help you with questions about:
• Company policies and procedures
• Employee benefits and insurance
• Payroll and compensation matters
• Time off and leave policies
• Workplace guidelines and safety protocols

Please feel free to rephrase your question or ask about any of these topics."""
                
                if is_first_message and supabase_client:
                    user_profile = self.get_user_profile(email, supabase_client)
                    welcome = self.generate_welcome_message(user_profile)
                    return f"{welcome}\n\n{fallback_message}"
                return fallback_message
            
            # Create minimal context
            context = ""
            for item in relevant_content:
                # Limit context size for speed
                content_preview = item['content'][:800] + "..." if len(item['content']) > 800 else item['content']
                context += f"From {item['source']}: {content_preview}\n\n"
                print(f"🎯 Using: {item['source']}")
              # Original prompt without keyword filtering
            prompt = f"""You are an AI HR Assistant providing automated support. Based on the company information below, answer the employee question in a helpful and professional manner.

Question: {question}

Company Information:
{context}

Instructions:
- Provide a clear, direct answer based on the company information
- Be helpful and professional
- Reference the source document when appropriate
- Keep the response concise (2-4 sentences)
- Act as an automated HR system, not as a human representative
- For complex or sensitive matters, suggest contacting HR directly at {self.hr_contact_email}

Answer:"""

            response = self.llm.invoke(prompt)
            
            # Generate URLs for referenced documents (happens in parallel with response processing)
            referenced_documents = []
            unique_files = set()
            
            for item in relevant_content:
                file_path = item.get('file_path')
                if file_path and file_path not in unique_files:
                    # Cache hit = instant, cache miss = minimal delay
                    signed_url = self.get_cached_signed_url(file_path, get_signed_url)
                    if signed_url:
                        referenced_documents.append({
                            'name': item['source'],
                            'url': signed_url,
                            'type': item.get('doc_type', 'document')
                        })
                        unique_files.add(file_path)
            
            # Start with the response content
            final_response = response.content
            
            # Add reference document links if any were found
            if referenced_documents:
                final_response += "\n\n📎 **Reference Documents:**"
                for doc in referenced_documents:
                    if doc.get('url'):  # Only add link if URL exists
                        final_response += f"\n [{doc['name']}]({doc['url']})"
                    else:  # If no URL, just show the document name
                        final_response += f"\n {doc['name']} (Document available through HR)"
              # Add HR contact info footer for complex queries
            final_response += f"\n\n---\n*For additional assistance, contact HR at {self.hr_contact_email} or {self.hr_phone}*"
            
            if is_first_message and supabase_client:
                user_profile = self.get_user_profile(email, supabase_client)
                welcome = self.generate_welcome_message(user_profile)
                final_response = f"{welcome}\n\n{final_response}"
            
            total_time = time.time() - start_time
            print(f"⚡ Response time: {total_time:.1f}s")
            
            return final_response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"""I'm experiencing technical difficulties at the moment. Please try again in a moment.

**For immediate assistance, please contact our HR team:**
📧 Email: {self.hr_contact_email}
📞 Phone: {self.hr_phone}
🕒 Business Hours: Monday-Friday, 9:00 AM - 5:00 PM EST"""

    def get_user_profile(self, email: str, supabase_client):
        """Get user profile for personalized greeting"""
        try:
            response = supabase_client.table("user_profiles") \
                .select("full_name, role, department") \
                .eq("email", email) \
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            return None
    
    def generate_welcome_message(self, user_profile: Dict = None) -> str:
        """Generate personalized welcome message"""
        if user_profile and user_profile.get('full_name'):
            name = user_profile['full_name'].split()[0]  # First name
            role = user_profile.get('role', '')
            dept = user_profile.get('department', '')
            
            if role and dept:
                return f"Hello {name}! 👋 I'm your AI HR Assistant. As a {role} in {dept}, I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"
            else:
                return f"Hello {name}! 👋 I'm your AI HR Assistant. I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"
        else:
            return "Hello! 👋 I'm your AI HR Assistant. I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"
