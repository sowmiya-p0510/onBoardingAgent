import os
import requests
from typing import List, Dict, Any, Optional
import PyPDF2
from io import BytesIO
import chromadb
import hashlib
import time
import threading
from datetime import datetime
import re

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

        # Track user sessions for personalized greetings
        self.user_sessions = {}

        # 🆕 STEP 1: Unlimited in-memory chat history storage
        self.chat_history = {}  # {email: [{'role': 'user'/'assistant', 'content': 'message', 'timestamp': datetime}, ...]}
        self.chat_history_lock = threading.Lock()  # Thread safety for concurrent access

        # 🆕 NEW: Track suggested follow-up actions per user to prevent repetition
        self.user_suggested_actions = {}  # {email: set(action_text, ...)} - tracks all suggested actions per user
        self.user_suggested_actions_lock = threading.Lock()  # Thread safety for concurrent access

        # Configuration for follow-up actions (no length constraints for natural conversation)
        self.max_tracked_actions = 50  # Maximum actions to track per user (to prevent memory bloat)

        # URL cache for fast document reference generation
        self.url_cache = {}  # {file_path: {'url': signed_url, 'expires_at': timestamp}}
        self.url_cache_duration = 3000  # 50 minutes (safe buffer from 1 hour expiration)

        # Company name for consistent branding
        self.company_name = "Fusefy"

        # Access portal URLs mapping for system access requests
        self.access_portals = {
            "slack": {
                "url": "https://portal.fusefy.ai/slack",
                "description": "Company Slack workspace for team communication",
                "keywords": ["slack", "chat", "messaging", "communication", "team chat", "channels"]
            },
            "github": {
                "url": "https://portal.fusefy.ai/git",
                "description": "GitHub repository access for code collaboration",
                "keywords": ["github", "git", "code", "repository", "version control", "development", "repo"]
            },
            "system": {
                "url": "https://portal.fusefy.ai/system",
                "description": "System access portal for IT resources",
                "keywords": ["system access", "system", "IT resources", "infrastructure", "server access"]
            },
            "portal": {
                "url": "https://portal.fusefy.ai/dashboard",
                "description": "Main company portal and dashboard",
                "keywords": ["portal", "dashboard", "main portal", "company portal", "home portal"]
            },
            "email": {
                "url": "https://portal.fusefy.ai/email",
                "description": "Company email system access",
                "keywords": ["email", "mail", "outlook", "company email", "mailbox"]
            },
            "vpn": {
                "url": "https://portal.fusefy.ai/vpn",
                "description": "VPN access for secure remote connections",
                "keywords": ["vpn", "remote access", "secure connection", "network access", "virtual private network"]
            },
            "timesheet": {
                "url": "https://portal.fusefy.ai/timesheet",
                "description": "Time tracking and timesheet management",
                "keywords": ["timesheet", "time tracking", "hours", "attendance", "clock in", "clock out"]
            },
            "jira": {
                "url": "https://portal.fusefy.ai/jira",
                "description": "Project management and issue tracking",
                "keywords": ["jira", "project management", "tickets", "issues", "project tracking", "bug tracking"]
            },
            "confluence": {
                "url": "https://portal.fusefy.ai/confluence",
                "description": "Company wiki and documentation platform",
                "keywords": ["confluence", "wiki", "documentation", "knowledge base", "docs"]
            },
            "office365": {
                "url": "https://portal.fusefy.ai/office365",
                "description": "Microsoft Office 365 suite access",
                "keywords": ["office 365", "office", "microsoft", "word", "excel", "powerpoint", "teams"]
            },
            "aws": {
                "url": "https://portal.fusefy.ai/aws",
                "description": "AWS cloud services access",
                "keywords": ["aws", "amazon web services", "cloud", "ec2", "s3", "lambda"]
            },
            "database": {
                "url": "https://portal.fusefy.ai/database",
                "description": "Database access and management tools",
                "keywords": ["database", "db", "sql", "mysql", "postgresql", "data access"]
            }
        }

        # Folder definitions for better LLM understanding
        self.folder_definitions = {
            "onboarding agent/benefits/": {
                "name": "Employee Benefits",
                "description": "Comprehensive information about employee compensation packages, health insurance, retirement plans, paid time off policies, and other benefit programs available to employees.",
                "content_type": "Benefits and compensation information",
                "use_cases": ["health insurance questions", "retirement planning", "leave policies", "benefit enrollment", "compensation packages"]
            },
            "onboarding agent/expense/": {
                "name": "Expense Management",
                "description": "Policies and procedures for business expense reporting, reimbursement processes, travel guidelines, and expense approval workflows.",
                "content_type": "Expense and reimbursement policies",
                "use_cases": ["expense reporting", "travel reimbursement", "business expenses", "receipt requirements", "approval processes"]
            },
            "onboarding agent/onboarding/": {
                "name": "Mandatory Onboarding Documents",
                "description": "Essential documents and procedures that all new employees must review, acknowledge, and complete during their onboarding process. These are mandatory requirements for all new hires.",
                "content_type": "Mandatory onboarding requirements and acknowledgment documents",
                "use_cases": ["new employee requirements", "mandatory training", "document acknowledgment", "onboarding checklist", "compliance requirements"],
                "mandatory": True,
                "priority": "high"
            },
            "onboarding agent/policy/": {
                "name": "Company Policies",
                "description": "Official company policies covering workplace conduct, compliance requirements, security protocols, and organizational guidelines that all employees must follow.",
                "content_type": "Corporate policies and compliance documentation",
                "use_cases": ["code of conduct", "security policies", "workplace guidelines", "compliance requirements", "disciplinary procedures"]
            }        
        }

        print("✅ Agent initialized - ready for preloading")

    # 🆕 STEP 2: Chat History Management Methods
    def add_to_chat_history(self, user_email: str, role: str, content: str) -> None:
        """Add a message to the user's chat history in memory"""
        with self.chat_history_lock:
            if user_email not in self.chat_history:
                self.chat_history[user_email] = []

            self.chat_history[user_email].append({
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now()
            })

            print(f"💬 Added {role} message to chat history for {user_email} (total: {len(self.chat_history[user_email])})")

    def get_chat_history(self, user_email: str) -> List[Dict]:
        """Retrieve the full chat history for a user"""
        with self.chat_history_lock:
            return self.chat_history.get(user_email, []).copy()  # Return a copy for thread safety

    def get_chat_history_for_llm(self, user_email: str) -> str:
        """Format chat history for inclusion in LLM prompt"""
        history = self.get_chat_history(user_email)

        if not history:
            return ""

        formatted_history = "\n\n🔄 **Previous Conversation History:**\n"
        for message in history:
            role_label = "Employee" if message['role'] == 'user' else "HR Assistant"
            formatted_history += f"{role_label}: {message['content']}\n\n"

        return formatted_history

    def clear_chat_history(self, user_email: str) -> bool:
        """Clear chat history for a specific user (utility method)"""
        with self.chat_history_lock:
            if user_email in self.chat_history:
                conversations_count = len(self.chat_history[user_email])
                del self.chat_history[user_email]
                print(f"🗑️ Cleared {conversations_count} conversations for {user_email}")
                return True
            return False

    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get statistics about chat history storage (utility method)"""
        with self.chat_history_lock:
            total_users = len(self.chat_history)
            total_messages = sum(len(history) for history in self.chat_history.values())

            user_stats = {}
            for email, history in self.chat_history.items():
                user_stats[email] = {
                    'message_count': len(history),
                    'last_message': history[-1]['timestamp'].isoformat() if history else None
                }

            return {
                'total_users': total_users,
                'total_messages': total_messages,
                'user_statistics': user_stats
            }

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
                role="Fusefy HR Assistant",
                goal="Provide quick, accurate answers about Fusefy policies and procedures",
                backstory=f"You are a helpful HR assistant for {self.company_name}, providing support to employees with company policies, benefits, and procedures.",
                verbose=False,
                allow_delegation=False,                
                llm=self.llm
            )
        return self._agent

    def get_cached_signed_url(self, file_path: str, get_signed_url) -> str:
        """Lightning-fast URL generation with intelligent caching"""
        current_time = time.time()

        print(f"🔗 URL request for: {file_path}")

        # Check cache first (microsecond lookup)
        if file_path in self.url_cache:
            cache_entry = self.url_cache[file_path]
            if current_time < cache_entry['expires_at']:
                print(f"🔗 Cache hit for: {file_path}")
                return cache_entry['url']  # Instant return from cache

        # Generate new URL only if needed
        try:
            print(f"🔗 Generating new URL for: {file_path}")
            new_url = get_signed_url(file_path, expiration=3600)  # 1 hour
            if new_url:
                self.url_cache[file_path] = {
                    'url': new_url,
                    'expires_at': current_time + self.url_cache_duration
                }
                print(f"🔗 Successfully generated URL for: {file_path}")
                return new_url
            else:
                print(f"🔗 get_signed_url returned empty/None for: {file_path}")
        except Exception as e:
            print(f"🔗 URL generation error for {file_path}: {e}")

        print(f"🔗 Failed to generate URL for: {file_path}")
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
            for doc_name, (content, file_path, folder_info) in documents.items():  # Unpack content, file path, and folder info
                chunks = self.chunk_text_simple(content)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)

                    # Improved document type classification based on folder path
                    doc_type = "document"  # default
                    is_mandatory = False

                    if "onboarding agent/policy/" in file_path.lower():
                        doc_type = "policy"
                    elif "onboarding agent/benefit" in file_path.lower():
                        doc_type = "benefit"
                    elif "onboarding agent/expense/" in file_path.lower():
                        doc_type = "expense"
                    elif "onboarding agent/onboarding/" in file_path.lower():
                        doc_type = "onboarding"
                        is_mandatory = True  # Mark onboarding documents as mandatory

                    all_metadatas.append({
                        "source": doc_name,
                        "file_path": file_path,  # Store the original GCS file path
                        "chunk_index": i,
                        "doc_type": doc_type,
                        "folder_name": folder_info.get('name', 'Unknown'),
                        "folder_description": folder_info.get('description', ''),
                        "content_type": folder_info.get('content_type', ''),
                        "is_mandatory": is_mandatory,
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

    def load_documents_parallel(self, list_files_in_folder, get_signed_url, check_file_exists) -> Dict[str, tuple]:
        """Load documents and store content with file paths and folder context"""
        documents = {}

        # Updated folder structure with parent "onboarding agent/" folder
        for folder in ["onboarding agent/benefits/", "onboarding agent/expense/", "onboarding agent/onboarding/", "onboarding agent/policy/"]:
            try:
                files = list_files_in_folder(folder)
                folder_info = self.folder_definitions.get(folder, {})
                print(f"📁 Processing {len(files)} files from {folder_info.get('name', folder)}")

                for file_path in files:
                    if file_path.endswith('.pdf') and check_file_exists(file_path):
                        signed_url = get_signed_url(file_path)
                        if signed_url:
                            content = self.extract_text_from_pdf_fast(signed_url)
                            if content:
                                file_name = file_path.split('/')[-1].replace('.pdf', '')
                                # Store content, file path, and folder context
                                documents[file_name] = (content, file_path, folder_info)
                                print(f"✅ Loaded: {file_name} ({folder_info.get('name', 'Unknown')})")

                        # Small delay to be gentle on GCS
                        time.sleep(0.1)

            except Exception as e:
                print(f"❌ Error loading from {folder}: {e}")

        return documents

    def search_relevant_content_fast(self, question: str) -> List[Dict]:
        """Fast search with minimal results and folder context"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=3,  # Increased from 2 to 3 for better coverage
                include=['documents', 'metadatas']            )

            relevant_content = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]

                print(f"🔍 Search result {i}: {metadata.get('source')} - file_path: {metadata.get('file_path')}")

                relevant_content.append({
                    'content': doc,
                    'source': metadata['source'],
                    'file_path': metadata.get('file_path'),
                    'doc_type': metadata.get('doc_type'),
                    'folder_name': metadata.get('folder_name'),
                    'folder_description': metadata.get('folder_description'),
                    'content_type': metadata.get('content_type'),
                    'is_mandatory': metadata.get('is_mandatory', False)
                })

            print(f"🔍 Total relevant content items: {len(relevant_content)}")
            return relevant_content

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def normalize_region(self, region_raw: str) -> str:
        """
        Normalize any detected region/city/country to a canonical region string for LLM use.
        Only returns: 'asia', 'africa', 'america', 'australia'. Defaults to 'asia'.
        """
        region_raw = (region_raw or "").strip().lower()
        canonical_regions = {"asia", "africa", "america", "australia"}
        # Direct match
        if region_raw in canonical_regions:
            return region_raw
        # Map common countries/cities to regions
        region_map = {
            # Asia
            "india": "asia", "chennai": "asia", "singapore": "asia", "japan": "asia", "china": "asia", "malaysia": "asia", "bangalore": "asia", "delhi": "asia", "mumbai": "asia", "asia": "asia",
            # Africa
            "nigeria": "africa", "south africa": "africa", "kenya": "africa", "africa": "africa",
            # America
            "usa": "america", "united states": "america", "canada": "america", "america": "america", "brazil": "america", "mexico": "america",
            # Australia
            "australia": "australia", "sydney": "australia", "melbourne": "australia"
        }
        for key, val in region_map.items():
            if key in region_raw:
                return val
        # Fallback
        return "asia"

    def get_user_profile(self, email: str, supabase_client, ip_address: str):
        """Get complete user profile for comprehensive personalization"""
        try:
            response = supabase_client.table("user_profiles") \
                .select("*") \
                .eq("email", email) \
                .execute()
            if response.data and len(response.data) > 0:
                user_profile = response.data[0]
                detected_region = self.detect_region_from_ip(ip_address)
                user_profile['region'] = self.normalize_region(detected_region)
                return user_profile
            detected_region = self.detect_region_from_ip(ip_address)
            return {'region': self.normalize_region(detected_region)}
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            detected_region = self.detect_region_from_ip(ip_address)
            return {'region': self.normalize_region(detected_region)}

    def process_chat_request(self, question: str, email: str, list_files_in_folder, get_signed_url, check_file_exists, supabase_client=None, ip_address: str = None) -> Dict[str, Any]:
        """Fast chat processing with unlimited in-memory chat history and region detection"""
        start_time = time.time()
        try:
            self.add_to_chat_history(email, 'user', question)
            is_first_message = email not in self.user_sessions
            user_profile = None
            if is_first_message and supabase_client:
                user_profile = self.get_user_profile(email, supabase_client, ip_address)
                self.user_sessions[email] = True
                greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
                if any(word in question.lower() for word in greeting_words) and len(question.split()) <= 3:
                    welcome_response = self.generate_welcome_message(user_profile)
                    self.add_to_chat_history(email, 'assistant', welcome_response)
                    return self.create_response_with_actions(welcome_response, question, email, user_profile)
            detected_access = self.detect_access_request(question)
            if detected_access["is_access_request"]:
                if not user_profile and supabase_client:
                    user_profile = self.get_user_profile(email, supabase_client, ip_address)
                access_response = self.generate_access_response(question, detected_access, user_profile)
                self.add_to_chat_history(email, 'assistant', access_response)
                return self.create_response_with_actions(access_response, question, email, user_profile)
            if self.detect_attestation_query(question):
                if not user_profile and supabase_client:
                    user_profile = self.get_user_profile(email, supabase_client, ip_address)
                if not self.is_initialized:
                    with self.initialization_lock:
                        if not self.is_initialized:
                            print("⚡ Quick vector store check...")
                            self.set_gcs_functions(list_files_in_folder, get_signed_url, check_file_exists)
                            if not self.initialize_vector_store_fast(list_files_in_folder, get_signed_url, check_file_exists):
                                init_response = "I'm currently initializing my knowledge base. Please try again in a moment, and I'll be ready to help you with your HR questions!"
                                self.add_to_chat_history(email, 'assistant', init_response)
                                return self.create_response_with_actions(init_response, question, email, user_profile)
                            self.is_initialized = True
                attestation_response = self.generate_enhanced_attestation_response(user_profile, get_signed_url, email, supabase_client)
                self.add_to_chat_history(email, 'assistant', attestation_response)
                return self.create_response_with_actions(attestation_response, question, email, user_profile)
            if not self.is_initialized:
                with self.initialization_lock:
                    if not self.is_initialized:
                        print("⚡ Quick vector store check...")
                        self.set_gcs_functions(list_files_in_folder, get_signed_url, check_file_exists)
                        if not self.initialize_vector_store_fast(list_files_in_folder, get_signed_url, check_file_exists):
                            init_response = "I'm currently initializing my knowledge base. Please try again in a moment, and I'll be ready to help you with your HR questions!"
                            self.add_to_chat_history(email, 'assistant', init_response)
                            return self.create_response_with_actions(init_response, question, email, user_profile)
                        self.is_initialized = True
            relevant_content = self.search_relevant_content_fast(question)
            # REGION FILTERING
            region = user_profile.get('region', 'global') if user_profile else 'global'
            region_content = []
            for item in relevant_content:
                # Check region in folder_name, file_path, or doc_type
                region_match = False
                for region_key in [region, region.capitalize()]:
                    if region_key in (item.get('folder_name', '').lower() + item.get('file_path', '').lower() + str(item.get('doc_type', '')).lower()):
                        region_match = True
                        break
                if region_match:
                    region_content.append(item)
            # Fallback: if no region-specific, use global or default
            if not region_content:
                for item in relevant_content:
                    if "global" in (item.get('folder_name', '').lower() + item.get('file_path', '').lower() + str(item.get('doc_type', '')).lower()):
                        region_content.append(item)
            # If still empty, fallback to all
            if not region_content:
                region_content = relevant_content
            # Build context only from region_content
            context = ""
            mandatory_docs = []
            folder_context = ""
            for item in region_content:
                content_preview = item['content'][:800] + "..." if len(item['content']) > 800 else item['content']
                folder_info = f"[{item.get('folder_name', 'Unknown')}] "
                if item.get('is_mandatory'):
                    folder_info += "(MANDATORY ONBOARDING DOCUMENT) "
                    mandatory_docs.append(item['source'])
                context += f"{folder_info}From {item['source']}: {content_preview}\n\n"
                print(f"🎯 Using: {item['source']} ({item.get('folder_name', 'Unknown')})")
            # Create folder definitions context
            if region_content:
                unique_folders = set()
                for item in region_content:
                    folder_path = None
                    if 'onboarding agent/benefits/' in item.get('file_path', ''):
                        folder_path = 'onboarding agent/benefits/'
                    elif 'onboarding agent/expense/' in item.get('file_path', ''):
                        folder_path = 'onboarding agent/expense/'
                    elif 'onboarding agent/onboarding/' in item.get('file_path', ''):
                        folder_path = 'onboarding agent/onboarding/'
                    elif 'onboarding agent/policy/' in item.get('file_path', ''):
                        folder_path = 'onboarding agent/policy/'

                    if folder_path and folder_path not in unique_folders:
                        unique_folders.add(folder_path)
                        folder_def = self.folder_definitions.get(folder_path, {})
                        folder_context += f"\n📁 {folder_def.get('name', 'Unknown')}: {folder_def.get('description', '')}"
            user_context = ""
            personalized_greeting = ""
            acknowledgment_context = ""

            if supabase_client:
                if not user_profile:
                    user_profile = self.get_user_profile(email, supabase_client, ip_address)

                if user_profile:
                    # Create comprehensive user context from all profile fields
                    user_context = f"""
Employee Profile Context:
- Name: {user_profile.get('full_name', 'N/A')}
- Email: {user_profile.get('email', 'N/A')}
- Role: {user_profile.get('role', 'N/A')}
- Department: {user_profile.get('department', 'N/A')}
- Manager: {user_profile.get('manager_name', 'N/A')} ({user_profile.get('manager_email', 'N/A')})
- Joining Date: {user_profile.get('joining_date', 'N/A')}
- Location: {user_profile.get('location', 'N/A')}
- Employment Type: {user_profile.get('employment_type', 'N/A')}
- User ID: {user_profile.get('user_id', 'N/A')}
"""
                    # Add personalized greeting for first-time users
                    if is_first_message:
                        name = user_profile.get('full_name', '').split()[0] if user_profile.get('full_name') else 'there'
                        role = user_profile.get('role', '')
                        dept = user_profile.get('department', '')

                        if role and dept:
                            personalized_greeting = f"Start your response with: 'Hello {name}! As a {role} in {dept}, here's what you need to know:'"
                        else:
                            personalized_greeting = f"Start your response with: 'Hello {name}! Here's what you need to know:'"

                # 🆕 Get document acknowledgment status for LLM context
                acknowledgment_context = self.format_acknowledgment_context_for_llm(email, supabase_client, get_signed_url)

            mandatory_notice = ""
            if mandatory_docs:                
                mandatory_notice = f"""
⚠️ IMPORTANT: The following documents contain MANDATORY ONBOARDING REQUIREMENTS that all new employees must review and acknowledge:
{', '.join(mandatory_docs)}
"""

            # 🆕 STEP 9: Get and include full chat history in the prompt
            chat_history_context = self.get_chat_history_for_llm(email)

            prompt = f"""You are an AI HR Assistant for {self.company_name} providing comprehensive support to employees. Based on the company information, folder context, employee profile, and conversation history below, provide a detailed and thorough answer to the employee's current question.

IMPORTANT INSTRUCTIONS:
- ONLY provide the answer for the user's region: {region.upper()} (must be one of: ASIA, AFRICA, AMERICA, AUSTRALIA).
- If the detected region is not one of these, always map it to the closest canonical region (e.g., 'Chennai', 'India', 'Singapore' → 'ASIA') and answer for that region.
- DO NOT use city or country names in your answer; always refer to the region only (e.g., 'Asia', not 'Chennai' or 'India').
- Do NOT include global or other region details in the main answer.
- If no content is found for the region, say so and offer to show another region.
- You represent {self.company_name} - always refer to the company as "{self.company_name}" when mentioning the organization.
- Consider the full conversation history to provide contextual and relevant responses.
- Provide direct, helpful responses without formal closings like "Best regards," "Sincerely," or "Thank you for your question".
- Keep responses conversational and professional but not overly formal.
- End responses with practical information or next steps, not pleasantries.

Current Question: {question}

{chat_history_context}

{user_context}

Document Folder Definitions:{folder_context}

{mandatory_notice}

{acknowledgment_context}

Company Information:
{context}

Instructions:
- {personalized_greeting if personalized_greeting else 'Provide a comprehensive, detailed answer based on the company information'}
- Always refer to the organization as "{self.company_name}" when mentioning the company
- Use the employee profile context to personalize your response when relevant
- Pay special attention to folder context - understand what type of information each folder contains
- For MANDATORY ONBOARDING documents, emphasize that these are required for all new employees
- When discussing mandatory onboarding documents, ALWAYS reference the user's acknowledgment status
- Provide specific counts of acknowledged vs pending documents when relevant
- Tailor information based on their role, department, employment type, and seniority
- Consider their joining date for tenure-based benefits or policies
- Reference their manager when escalation or approval processes are mentioned
- Adapt explanations based on their location (remote vs office policies)
- Include all relevant details, procedures, eligibility criteria, and requirements
- Break down complex information into clear sections or bullet points
- Explain the reasoning behind policies when possible
- Include specific examples or scenarios where applicable, especially relevant to their role/department
- Reference multiple source documents when they provide related information
- Use clear headings or sections to organize detailed information
- Provide step-by-step procedures when explaining processes
- Include important deadlines, timeframes, or restrictions
- Mention any prerequisites or conditions that apply
- Be thorough and educational in your response
- DO NOT include HR contact information in your response as it will be added separately
- Focus on the policy/document content without repeating contact details from documents
- For complex matters, still suggest contacting HR for personalized guidance but do not include specific contact details
- DO NOT end with formal closings like "Best regards," "Sincerely," "Thank you," or similar phrases
- End with practical next steps or relevant information instead of pleasantries

Provide a complete and informative response that fully addresses the question with personalized context and proper folder categorization.

FOLLOW-UP SUGGESTIONS INSTRUCTIONS:
- After your comprehensive answer, generate exactly two topic suggestions for what the user might want to explore next
- Format each suggestion as a direct topic statement, not a question (e.g., "Explain the code of conduct policy" instead of "Would you like to know about the code of conduct?")
- Make suggestions specific and directly related to the information you just provided
- If discussing region-specific information, ONE suggestion can be about another region's policies
- The other suggestion should explore a different but related aspect of the current topic
- Suggestions should be concise (5-8 words) and actionable
- Do NOT use "Would you like to..." or similar question phrasing
- Each suggestion should be a direct topic statement the user could ask about next
"""

            response = self.llm.invoke(prompt)

            # Generate URLs for referenced documents (with support for pre-generated URLs)
            referenced_documents = []
            unique_files = set()

            print(f"🔗 Processing {len(relevant_content)} items for reference documents...")

            for item in relevant_content:
                file_path = item.get('file_path')
                print(f"🔗 Processing item: {item.get('source')} with file_path: {file_path}")

                if file_path and file_path not in unique_files:
                    # Check if URL is already generated (for attestation queries)
                    signed_url = item.get('signed_url')
                    if not signed_url:
                        print(f"🔗 Generating signed URL for: {file_path}")
                        # Generate URL if not already available
                        signed_url = self.get_cached_signed_url(file_path, get_signed_url)
                        print(f"🔗 Generated URL: {signed_url[:50]}..." if signed_url else "🔗 Failed to generate URL")

                    if signed_url:
                        referenced_documents.append({
                            'name': item['source'],
                            'url': signed_url,
                            'type': item.get('doc_type', 'document')
                        })
                        unique_files.add(file_path)
                        print(f"🔗 ✅ Added reference document: {item['source']}")
                    else:
                        print(f"🔗 ❌ No URL for document: {item['source']}")

            print(f"🔗 Total reference documents: {len(referenced_documents)}")

            # Start with the response content
            final_response = response.content

            # Add reference document links if any were found
            if referenced_documents:
                print(f"🔗 Adding {len(referenced_documents)} reference documents to response")
                final_response += "\n\n📎 **Reference Documents:**"
                for doc in referenced_documents:
                    if doc.get('url'):  # Only add link if URL exists
                        final_response += f"\n• [{doc['name']}]({doc['url']})"
                    else:  # If no URL, just show the document name
                        final_response += f"\n• {doc['name']} (Document available through HR)"
            elif relevant_content:
                # Fallback: show document names even without URLs
                print("🔗 No URLs generated, showing document names as fallback")
                final_response += "\n\n📎 **Reference Documents:**"
                unique_sources = set()
                for item in relevant_content:
                    source = item.get('source')
                    if source and source not in unique_sources:
                        final_response += f"\n• {source} (Document available through HR)"
                        unique_sources.add(source)
            else:
                print("🔗 No reference documents found or generated")

            # 🆕 STEP 10: Add assistant response to chat history
            self.add_to_chat_history(email, 'assistant', final_response)

            total_time = time.time() - start_time
            print(f"⚡ Response time: {total_time:.1f}s")

            return self.create_response_with_actions(final_response, question, email, user_profile)

        except Exception as e:
            print(f"❌ Error: {e}")
            error_response = f"""I'm experiencing technical difficulties at the moment. Please try again in a moment.

If the issue persists, you may want to contact your HR team for assistance."""

            # 🆕 Add error response to chat history
            self.add_to_chat_history(email, 'assistant', error_response)
            return self.create_response_with_actions(error_response, question, email, user_profile)

    def detect_region_from_ip(self, ip_address: str) -> str:
        """Method to detect user's region via IP"""
        try:
            response = requests.get(f'https://ipapi.co/{ip_address}/json/')
            response.raise_for_status()
            data = response.json()
            return data.get('country_name', 'Unknown')
        except Exception as e:
            print(f"❌ Error detecting region from IP: {e}")
            return 'Unknown'

    def get_user_profile(self, email: str, supabase_client, ip_address: str):
        """Get complete user profile for comprehensive personalization"""
        try:
            response = supabase_client.table("user_profiles") \
                .select("*") \
                .eq("email", email) \
                .execute()
            if response.data and len(response.data) > 0:
                user_profile = response.data[0]
                detected_region = self.detect_region_from_ip(ip_address)
                user_profile['region'] = self.normalize_region(detected_region)
                return user_profile
            detected_region = self.detect_region_from_ip(ip_address)
            return {'region': self.normalize_region(detected_region)}
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            detected_region = self.detect_region_from_ip(ip_address)
            return {'region': self.normalize_region(detected_region)}

    def get_user_document_acknowledgments(self, email: str, supabase_client) -> Dict[str, Any]:
        """Get user's document acknowledgment status from Supabase"""
        try:
            response = supabase_client.table("document_acknowledgments") \
                .select("*") \
                .eq("email", email) \
                .order("acknowledged_at", desc=True) \
                .execute()

            acknowledgments = {}
            acknowledgment_details = []

            if response.data:
                for ack in response.data:
                    doc_name = ack['document_name']
                    acknowledgments[doc_name] = {
                        'acknowledged': True,
                        'acknowledged_at': ack['acknowledged_at'],
                        'acknowledgment_id': ack['id']
                    }
                    acknowledgment_details.append({
                        'document_name': doc_name,
                        'acknowledged_at': ack['acknowledged_at'],
                        'acknowledgment_id': ack['id']
                    })

            return {
                'acknowledgments': acknowledgments,
                'acknowledgment_details': acknowledgment_details,
                'total_acknowledged': len(acknowledgments)
            }
        except Exception as e:
            print(f"Error fetching document acknowledgments: {e}")
            return {
                'acknowledgments': {},
                'acknowledgment_details': [],
                'total_acknowledged': 0
            }

    def get_document_status_summary(self, email: str, supabase_client, get_signed_url) -> Dict[str, Any]:
        """Get comprehensive document status including acknowledged vs pending documents"""
        try:
            # Get all mandatory documents
            all_mandatory_docs = self.get_all_mandatory_documents(get_signed_url)

            # Get user's acknowledgments
            user_acknowledgments = self.get_user_document_acknowledgments(email, supabase_client)

            acknowledged_docs = []
            pending_docs = []

            # Normalize acknowledgments for accurate matching
            normalized_acknowledgments = {name.strip().lower(): info for name, info in user_acknowledgments['acknowledgments'].items()}

            for doc in all_mandatory_docs:
                doc_name_normalized = doc['source'].strip().lower()
                if doc_name_normalized in normalized_acknowledgments:
                    # Document is acknowledged
                    ack_info = normalized_acknowledgments[doc_name_normalized]
                    acknowledged_docs.append({
                        **doc,
                        'acknowledged': True,
                        'acknowledged_at': ack_info['acknowledged_at'],
                        'acknowledgment_id': ack_info['acknowledgment_id']
                    })
                else:
                    # Document is pending acknowledgment
                    pending_docs.append({
                        **doc,
                        'acknowledged': False,
                        'status': 'pending_acknowledgment'
                    })

            return {
                'all_documents': all_mandatory_docs,
                'acknowledged_documents': acknowledged_docs,
                'pending_documents': pending_docs,
                'total_documents': len(all_mandatory_docs),
                'total_acknowledged': len(acknowledged_docs),
                'total_pending': len(pending_docs),
                'completion_percentage': (len(acknowledged_docs) / len(all_mandatory_docs) * 100) if all_mandatory_docs else 100,
                'user_acknowledgments': user_acknowledgments
            }
        except Exception as e:
            print(f"Error getting document status summary: {e}")
            return {
                'all_documents': [],
                'acknowledged_documents': [],
                'pending_documents': [],
                'total_documents': 0,
                'total_acknowledged': 0,
                'total_pending': 0,
                'completion_percentage': 0,
                'user_acknowledgments': {'acknowledgments': {}, 'acknowledgment_details': [], 'total_acknowledged': 0}
            }

    def format_acknowledgment_context_for_llm(self, email: str, supabase_client, get_signed_url) -> str:
        """Format document acknowledgment status for LLM context"""
        try:
            doc_status = self.get_document_status_summary(email, supabase_client, get_signed_url)

            if doc_status['total_documents'] == 0:
                return ""

            context = f"\n\n📋 **DOCUMENT ACKNOWLEDGMENT STATUS FOR {email.upper()}:**\n"
            context += f"📊 Progress: {doc_status['total_acknowledged']}/{doc_status['total_documents']} documents acknowledged ({doc_status['completion_percentage']:.1f}% complete)\n\n"

            if doc_status['acknowledged_documents']:
                context += "✅ **ACKNOWLEDGED DOCUMENTS:**\n"
                for doc in doc_status['acknowledged_documents']:
                    ack_date = doc['acknowledged_at'][:10] if doc['acknowledged_at'] else 'Unknown'
                    context += f"   • {doc['source']} (acknowledged on {ack_date})\n"
                context += "\n"

            if doc_status['pending_documents']:
                context += "⏳ **PENDING ACKNOWLEDGMENT (REQUIRED):**\n"
                for doc in doc_status['pending_documents']:
                    context += f"   • {doc['source']} - REQUIRES ACKNOWLEDGMENT\n"
                context += "\n"

            if doc_status['total_pending'] > 0:
                context += f"⚠️ **IMPORTANT:** {doc_status['total_pending']} mandatory document(s) still require acknowledgment to complete onboarding.\n"
            else:
                context += "🎉 **ALL MANDATORY DOCUMENTS ACKNOWLEDGED:** Onboarding documentation requirements are complete!\n"

            return context

        except Exception as e:
            print(f"Error formatting acknowledgment context: {e}")
            return ""

    def generate_enhanced_attestation_response(self, user_profile: Dict = None, get_signed_url=None, email: str = "", supabase_client=None) -> str:
        """Generate enhanced attestation response with acknowledgment status"""

        # Get document status summary
        if supabase_client:
            doc_status = self.get_document_status_summary(email, supabase_client, get_signed_url)
        else:
            # Fallback to basic mandatory docs if no Supabase client
            mandatory_docs = self.get_all_mandatory_documents(get_signed_url)
            doc_status = {
                'all_documents': mandatory_docs,
                'acknowledged_documents': [],
                'pending_documents': mandatory_docs,
                'total_documents': len(mandatory_docs),
                'total_acknowledged': 0,
                'total_pending': len(mandatory_docs),
                'completion_percentage': 0
            }

        # Personalized greeting
        name = ""
        if user_profile and user_profile.get('full_name'):
            name = f" {user_profile['full_name'].split()[0]}"
        else:
            name = ""

        # Status-aware greeting
        if doc_status['total_pending'] == 0:
            greeting = f"Hello{name}! 🎉 Excellent news - you've completed all mandatory onboarding document acknowledgments!"
        else:
            greeting = f"Hello{name}! Here's your current onboarding document status:"

        # Build status summary
        status_summary = f"""
**📊 Onboarding Progress: {doc_status['total_acknowledged']}/{doc_status['total_documents']} documents ({doc_status['completion_percentage']:.1f}% complete)**
"""

        # Build acknowledged documents section
        acknowledged_section = ""
        if doc_status['acknowledged_documents']:
            acknowledged_section = "\n**✅ Completed Acknowledgments:**\n"
            for doc in doc_status['acknowledged_documents']:
                ack_date = doc['acknowledged_at'][:10] if doc['acknowledged_at'] else 'Unknown'
                acknowledged_section += f"• {doc['source']} ✓ (acknowledged {ack_date})\n"

        # Build pending documents section
        pending_section = ""
        pending_links = ""
        if doc_status['pending_documents']:
            pending_section = "\n**⏳ Pending Acknowledgments (Required):**\n"
            pending_links = "\n**📎 Documents Requiring Acknowledgment:**\n"

            for i, doc in enumerate(doc_status['pending_documents'], 1):
                pending_section += f"{i}. {doc['source']} - **REQUIRES ACKNOWLEDGMENT**\n"
                if doc.get('signed_url'):
                    pending_links += f"[📄 {doc['source']}]({doc['signed_url']})\n"

        # Next steps
        next_steps = ""
        if doc_status['total_pending'] > 0:
            next_steps = f"""
**Next Steps:**
1. Review each pending document carefully
2. Contact HR if you have questions about any content
3. Complete acknowledgments to finish your onboarding
4. Keep copies for your records

**Important:** {doc_status['total_pending']} document(s) still require acknowledgment to complete your {self.company_name} onboarding process."""
        else:
            next_steps = f"""
**🎉 All Done!** You've successfully acknowledged all mandatory onboarding documents for {self.company_name}. Your onboarding documentation requirements are complete!

If you need to reference any documents later, contact your HR department."""

        response = f"""{greeting}

{status_summary}{acknowledged_section}{pending_section}{pending_links}{next_steps}"""

        return response

    def generate_welcome_message(self, user_profile: Dict = None) -> str:
        """Generate personalized welcome message"""
        if user_profile and user_profile.get('full_name'):
            name = user_profile['full_name'].split()[0]  # First name
            role = user_profile.get('role', '')
            dept = user_profile.get('department', '')

            if role and dept:
                return f"Hello {name}! 👋 I'm your AI HR Assistant for {self.company_name}. As a {role} in {dept}, I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"
            else:
                return f"Hello {name}! 👋 I'm your AI HR Assistant for {self.company_name}. I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"
        else:
            return f"Hello! 👋 I'm your AI HR Assistant for {self.company_name}. I'm here to help you with any questions about company policies, benefits, or procedures. What can I help you with today?"

    def detect_attestation_query(self, question: str) -> bool:
        """Detect if the question is asking about documents to attest/acknowledge"""
        attestation_keywords = [
            'attest', 'acknowledge', 'mandatory', 'onboarding', 'required documents',
            'need to sign', 'compliance documents', 'must review', 'acknowledgment',
            'documents to complete', 'new employee documents', 'required reading',
            'what documents', 'which documents', 'documents i need', 'docs to review',
            'mandatory documents', 'documents for onboarding', 'onboarding process',
            'documents to acknowledge', 'documents for new employees'
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in attestation_keywords)

    def get_all_mandatory_documents(self, get_signed_url) -> List[Dict]:
        """Get ALL mandatory onboarding documents for attestation purposes"""
        if not self.collection:
            return []

        try:
            # Get ALL documents marked as mandatory (onboarding folder)
            results = self.collection.get(
                where={"is_mandatory": True},
                include=['documents', 'metadatas']
            )

            mandatory_docs = []
            unique_sources = set()

            for i, metadata in enumerate(results['metadatas']):
                source = metadata['source']
                # Only include each document once (avoid duplicates from chunks)
                if source not in unique_sources:
                    unique_sources.add(source)

                    # Generate signed URL for each document
                    file_path = metadata.get('file_path')
                    signed_url = ""
                    if file_path:
                        signed_url = self.get_cached_signed_url(file_path, get_signed_url)

                    mandatory_docs.append({
                        'content': results['documents'][i] if i < len(results['documents']) else '',
                        'source': source,
                        'file_path': file_path,
                        'doc_type': metadata.get('doc_type'),
                        'folder_name': metadata.get('folder_name'),
                        'folder_description': metadata.get('folder_description'),
                        'content_type': metadata.get('content_type'),
                        'is_mandatory': True,
                        'signed_url': signed_url
                    })

            print(f"📋 Found {len(mandatory_docs)} mandatory documents for attestation")
            return mandatory_docs

        except Exception as e:
            print(f"❌ Error getting mandatory documents: {e}")
            return []

    def detect_access_request(self, question: str) -> Dict[str, Any]:
        """Detect if the question is asking for system access or portal information"""
        question_lower = question.lower()
        detected_portals = []

        # Check for general access keywords
        access_keywords = ["access", "login", "log in", "sign in", "portal", "how to get to", "link to", "url for"]
        is_access_related = any(keyword in question_lower for keyword in access_keywords)

        if is_access_related:
            for portal_key, portal_info in self.access_portals.items():
                for keyword in portal_info["keywords"]:
                    if keyword in question_lower:
                        detected_portals.append({
                            "name": portal_key,
                            "url": portal_info["url"],
                            "description": portal_info["description"]
                        })
                        break  # Found a match for this portal, move to next

        return {
            "is_access_request": len(detected_portals) > 0,
            "portals": detected_portals
        }

    def generate_access_response(self, question: str, detected_access: Dict, user_profile: Dict = None) -> str:
        """Generate response for access-related questions"""
        portals = detected_access["portals"]

        # Personalized greeting
        name = ""
        if user_profile and user_profile.get('full_name'):
            name = user_profile['full_name'].split()[0] + ", "

        if len(portals) == 1:
            portal = portals[0]
            return f"""Hi {name}here's the access information you need:

🔗 **{portal['description']}**
Access URL: [{portal['url']}]({portal['url']})

**Getting Started:**
1. Click the link above to access the portal
2. Use your {self.company_name} credentials to log in
3. If you encounter any login issues, contact IT support

**Need Help?**
- For login issues: Contact IT support
- For access permissions: Contact your manager or HR
- For account setup: Contact your HR department

If you need access to additional systems, please let me know and I'll provide the appropriate portal links."""

        else:
            # Multiple portals detected
            portal_list = ""
            for portal in portals:
                portal_list += f"- **{portal['description']}**: [{portal['url']}]({portal['url']})\n"

            return f"""Hi {name}here are the access portals you requested:

🔗 **System Access Portals:**

{portal_list}

**Getting Started:**
1. Click on any of the links above to access the respective portal
2. Use your {self.company_name} credentials to log in
3. If you encounter any login issues, contact IT support

**Need Help?**
- For login issues: Contact IT support
- For access permissions: Contact your manager or HR
- For account setup: Contact your HR department

If you need access to additional systems not listed above, please let me know and I'll help you find the right portal."""

    def generate_attestation_response(self, user_profile: Dict = None, get_signed_url=None) -> str:
        """Generate a clean, focused response for mandatory document attestation queries"""

        # Get mandatory documents
        mandatory_docs = self.get_all_mandatory_documents(get_signed_url)

        # Personalized greeting
        name = ""
        if user_profile and user_profile.get('full_name'):
            name = user_profile['full_name'].split()[0]
            role = user_profile.get('role', '')
            dept = user_profile.get('department', '')
            greeting = f"Hello {name}! "
            if role and dept:
                greeting += f"As a {role} in {dept}, here are the mandatory documents you need to acknowledge for your {self.company_name} onboarding process:"
            else:
                greeting += f"Here are the mandatory documents you need to acknowledge for your {self.company_name} onboarding process:"
        else:
            greeting = f"Here are the mandatory documents you need to acknowledge for your {self.company_name} onboarding process:"

        # Build clean document list
        doc_list = ""
        doc_links = ""

        for i, doc in enumerate(mandatory_docs, 1):
            doc_name = doc['source'].replace('_', ' ').title()
            doc_list += f"{i}. **{doc_name}**\n"

            # Add document link if available
            if doc.get('signed_url'):
                doc_links += f"📄 [{doc_name}]({doc['signed_url']})\n"

        response = f"""{greeting}

**📋 Mandatory Onboarding Documents:**

{doc_list}

**📎 Document Links:**
{doc_links}

**Next Steps:**
1. Review each document carefully
2. Contact HR if you have any questions about the content
3. Complete any required acknowledgments as instructed
4. Keep copies for your records

**Important:** These documents are mandatory for all new employees and must be acknowledged to complete your {self.company_name} onboarding process.

For questions about these documents, contact your HR department."""

        return response

    # Updated to generate more direct topic suggestions
    def generate_follow_up_actions(self, user_question: str, assistant_response: str, email: str, user_profile: Dict = None) -> List[str]:
        """Generate direct topic suggestions based on conversation context"""
        # Extract key topics from the conversation
        topics = self.extract_topics_from_conversation(user_question, assistant_response)

        # Get conversation history for context
        history = self.get_chat_history(email)

        # Build prompt for generating follow-up suggestions
        prompt = f"""Based on the following conversation, generate TWO direct topic suggestions that the user might want to explore next.

CURRENT QUESTION: {user_question}

ASSISTANT RESPONSE SUMMARY: {assistant_response[:500]}... (truncated)

CONVERSATION TOPICS: {', '.join(topics)}

USER PROFILE: {user_profile if user_profile else 'Not available'}

RULES FOR SUGGESTIONS:
1. Format as direct topic statements, NOT questions (e.g., "Explain the code of conduct policy" not "Would you like to know about the code of conduct?")
2. Make suggestions specific and directly related to the information just provided
3. If discussing region-specific information, ONE suggestion can be about another region's policies
4. The other suggestion should explore a different but related aspect of the current topic
5. Keep suggestions concise (5-8 words) and actionable
6. Do NOT use "Would you like to..." or similar question phrasing
7. Each suggestion should be a direct topic statement the user could ask about next
8. Suggestions should be phrased as commands or topics, not questions

Generate exactly TWO topic suggestions:"""

        try:
            # Generate follow-up suggestions
            response = self.llm.invoke(prompt)

            # Extract suggestions from response
            suggestions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('-') or line.startswith('•')):
                    # Clean up the suggestion
                    suggestion = re.sub(r'^[1-9][.)]|\-|\•|\*', '', line).strip()
                    if suggestion and len(suggestions) < 2:
                        suggestions.append(suggestion)
                elif line and len(suggestions) < 2 and len(line) < 100:  # Reasonable length for a suggestion
                    suggestions.append(line.strip())

            # Ensure we have exactly 2 suggestions
            while len(suggestions) < 2:
                if len(suggestions) == 0:
                    suggestions.append("Explain company code of conduct")
                else:
                    suggestions.append("Show HR contact information")

            return suggestions[:2]  # Return exactly 2 suggestions

        except Exception as e:
            print(f"Error generating follow-up suggestions: {e}")
            return [
                "Explain company code of conduct",
                "Show HR contact information"
            ]

    def extract_topics_from_conversation(self, user_question: str, assistant_response: str) -> List[str]:
        """Extract key topics from the conversation to inform follow-up suggestion generation"""
        combined_text = f"{user_question} {assistant_response}"
        combined_text_lower = combined_text.lower()

        # Define topic categories and their keywords
        topic_categories = {
            "benefits": ["benefits", "insurance", "health", "dental", "vision", "retirement", "401k", "pension", "pto", "vacation", "leave", "time off"],
            "policies": ["policy", "policies", "rules", "guidelines", "handbook", "code of conduct", "compliance", "regulations"],
            "expenses": ["expense", "reimbursement", "claim", "travel", "per diem", "receipt", "budget", "spending"],
            "onboarding": ["onboarding", "orientation", "new hire", "first day", "training", "setup", "equipment", "laptop"],
            "payroll": ["payroll", "salary", "compensation", "pay", "direct deposit", "tax", "deduction", "bonus", "raise"],
            "time_off": ["vacation", "sick leave", "pto", "time off", "holiday", "absence", "leave", "day off"],
            "performance": ["performance", "review", "evaluation", "feedback", "goals", "objectives", "kpi", "metrics"],
            "training": ["training", "development", "learning", "course", "certification", "skill", "workshop"],
            "remote_work": ["remote", "work from home", "wfh", "telecommute", "virtual", "hybrid", "flexible"],
            "hr_processes": ["process", "procedure", "workflow", "steps", "application", "form", "submit", "approval"],
            "regions": ["region", "asia", "africa", "america", "australia", "global", "country", "location"]
        }

        # Detect topics in the conversation
        detected_topics = []
        for topic, keywords in topic_categories.items():
            if any(keyword in combined_text_lower for keyword in keywords):
                detected_topics.append(topic)

        # Add region if mentioned
        for region in ["asia", "africa", "america", "australia"]:
            if region in combined_text_lower and "regions" not in detected_topics:
                detected_topics.append("regions")
                break

        return detected_topics if detected_topics else ["general"]

    # 🆕 STEP 11: Create helper method to format response with actions
    def create_response_with_actions(self, response_text: str, user_question: str, email: str, user_profile: Dict = None) -> Dict[str, Any]:
        """Create formatted response dictionary with direct topic suggestions"""
        print(f"🔍 Creating response with actions for: {email}")

        # First try to extract direct topic suggestions from the response
        # Look for patterns like "1. Topic suggestion" or "- Topic suggestion" at the end of the text
        suggestion_patterns = [
            r'\n\d+\.\s+([^\n?]+)(?=\n\d+\.|\Z)',  # Numbered list items without question marks
            r'\n-\s+([^\n?]+)(?=\n-|\Z)',          # Dash list items without question marks
            r'\n•\s+([^\n?]+)(?=\n•|\Z)'           # Bullet list items without question marks
        ]

        suggestions = []
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                if match.strip() and not match.strip().endswith('?'):
                    suggestions.append(match.strip())

        # If we found at least 2 suggestions, use them
        if len(suggestions) >= 2:
            actions = suggestions[:2]

            # Clean up the main response by removing these suggestions
            for suggestion in actions:
                response_text = re.sub(rf"\n[•\-\d\.]+\s+{re.escape(suggestion)}", "", response_text)

            # Clean up any remaining formatting artifacts
            response_text = re.sub(r'\n+\s*$', '', response_text)
            response_text = re.sub(r'(Follow-up Suggestions?:?\s*$|Topic Suggestions?:?\s*$)', '', response_text, flags=re.IGNORECASE)
        else:
            # Generate new suggestions if none or not enough were found
            actions = self.generate_follow_up_actions(user_question, response_text, email, user_profile)

        result = {
            "response": response_text.strip(),
            "actions": actions
        }

        print(f"🔍 Final result: response length={len(response_text)}, actions={actions}")
        return result

    def clear_user_suggested_actions(self, email: str):
        """Clear suggested actions history for a user (useful for new sessions or reset)"""
        with self.user_suggested_actions_lock:
            if email in self.user_suggested_actions:
                del self.user_suggested_actions[email]
                print(f"🧹 Cleared suggested actions history for user: {email}")

    def get_user_suggested_actions_count(self, email: str) -> int:
        """Get the count of tracked actions for a user (for monitoring/debugging)"""
        with self.user_suggested_actions_lock:
            return len(self.user_suggested_actions.get(email, set()))

    def get_all_suggested_actions_stats(self) -> Dict[str, Any]:
        """Get statistics about all users' suggested actions (for monitoring)"""
        with self.user_suggested_actions_lock:
            stats = {
                'total_users': len(self.user_suggested_actions),
                'total_unique_actions': sum(len(actions) for actions in self.user_suggested_actions.values()),
                'users_with_actions': {}
            }

            for email, actions in self.user_suggested_actions.items():
                stats['users_with_actions'][email] = len(actions)

            return stats

    def test_follow_up_generation(self, email: str = "test@example.com"):
        """Test method to debug follow-up action generation"""
        print("🧪 Testing follow-up action generation...")

        test_question = "Tell me about leave policies"
        test_response = "Here are the leave policies for Fusefy..."
        test_profile = {"full_name": "Test User", "role": "Developer"}

        actions = self.generate_follow_up_actions(test_question, test_response, email, test_profile)
        print(f"🧪 Generated actions: {actions}")
        return actions