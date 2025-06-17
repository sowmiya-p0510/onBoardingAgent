import os
import requests
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import PyPDF2
from io import BytesIO

class ChatAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        self.agent = Agent(
            role="HR Policy and Benefits Assistant",
            goal="Answer employee questions based on company policies and benefits documents",
            backstory="""You are an expert HR assistant with deep knowledge of company policies 
            and benefits. You help employees understand their benefits, policies, and procedures 
            by providing accurate information based on official company documents.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Cache for document content to avoid repeated downloads
        self.document_cache = {}
    
    def extract_text_from_pdf(self, pdf_url: str) -> str:
        """Extract text content from PDF using signed URL"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def load_documents(self, list_files_in_folder, get_signed_url, check_file_exists) -> Dict[str, str]:
        """Load and cache ALL policy and benefit documents"""
        if self.document_cache:
            return self.document_cache
        
        documents = {}
        
        # Load policy documents from "policies/" folder
        policy_folder = "policies/"
        try:
            policy_files = list_files_in_folder(policy_folder)
            print(f"Found {len(policy_files)} files in policies folder")
            
            for file_path in policy_files:
                if file_path.endswith('.pdf') and check_file_exists(file_path):
                    signed_url = get_signed_url(file_path)
                    content = self.extract_text_from_pdf(signed_url)
                    if content:
                        file_name = file_path.split('/')[-1].replace('.pdf', '')
                        documents[f"policy_{file_name}"] = content
                        print(f"Loaded policy document: {file_name}")
        except Exception as e:
            print(f"Error loading policy documents: {e}")
        
        # Load benefit documents from "benefits/" folder
        benefit_folder = "benefits/"
        try:
            benefit_files = list_files_in_folder(benefit_folder)
            print(f"Found {len(benefit_files)} files in benefits folder")
            
            for file_path in benefit_files:
                if file_path.endswith('.pdf') and check_file_exists(file_path):
                    signed_url = get_signed_url(file_path)
                    content = self.extract_text_from_pdf(signed_url)
                    if content:
                        file_name = file_path.split('/')[-1].replace('.pdf', '')
                        documents[f"benefit_{file_name}"] = content
                        print(f"Loaded benefit document: {file_name}")
        except Exception as e:
            print(f"Error loading benefit documents: {e}")
        
        self.document_cache = documents
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def process_chat_request(self, question: str, list_files_in_folder, get_signed_url, check_file_exists) -> str:
        """Process chat request and return response based on ALL documents"""
        try:
            # Load ALL documents from GCS bucket
            documents = self.load_documents(list_files_in_folder, get_signed_url, check_file_exists)
            
            if not documents:
                return "I'm sorry, I couldn't access the company documents at the moment. Please try again later."
            
            # Create context from ALL documents (not just keyword-matched ones)
            all_documents_text = ""
            for doc_name, content in documents.items():
                all_documents_text += f"\n\nDocument: {doc_name}\nContent: {content}\n"
            
            # Create task for the agent with ALL document content
            task = Task(
                description=f"""
                Based on the following company policy and benefits documents, answer this employee question: "{question}"
                
                ALL Company Documents Available:
                {all_documents_text}
                
                Instructions:
                1. Search through ALL the provided company documents to find relevant information
                2. Only answer based on the provided company documents
                3. Be helpful and accurate
                4. If the question cannot be answered from ANY of the documents, say: "I'm sorry, I couldn't find relevant information in our company policies and benefits documents to answer your question. Please contact HR directly for assistance."
                5. Provide specific details when available from any document
                6. Keep the response concise but comprehensive
                7. If information spans multiple documents, synthesize the information
                8. Mention which document(s) the information comes from when relevant
                """,
                agent=self.agent,
                expected_output="A helpful response based on ALL company policies and benefits documents"
            )
            
            # Execute the task
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            print(f"Error processing chat request: {e}")
            return "I'm experiencing technical difficulties. Please try again later or contact HR directly."
