from typing import List, Optional
from pydantic import BaseModel

class OnboardingFetchRequest(BaseModel):
    role: str
    email: str

class Document(BaseModel):
    title: str
    url: str
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None

class AcknowledgmentSummary(BaseModel):
    total_documents: int
    acknowledged_count: int
    pending_count: int

class OnboardingFetchResponse(BaseModel):
    success: bool
    message: str
    documents: List[Document] = []
    acknowledgment_summary: Optional[AcknowledgmentSummary] = None

class OnboardingAgent:
    def __init__(self, agent):
        """
        Initialize the Onboarding Agent with a pre-defined agent.

        Args:
            agent: A CrewAI Agent instance (not used in optimized version)
        """
        # Agent is no longer needed but kept for compatibility
        self.agent = agent

    def get_role_documents(self, role: str, supabase_client):
        """Get documents for a specific role from Supabase."""
        try:
            response = supabase_client.table("agents") \
                .select("*") \
                .eq("role", role) \
                .eq("agent_type", "onboarding") \
                .execute()

            if not response.data:
                return []

            agent_data = response.data[0]
            return agent_data.get("documents", [])
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def get_user_acknowledgments(self, email: str, supabase_client):
        """Get user's document acknowledgments from Supabase."""
        try:
            response = supabase_client.table("document_acknowledgments") \
                .select("document_name, acknowledged_at") \
                .eq("email", email) \
                .execute()
            
            # Store acknowledgments with original keys (no normalization)
            acknowledgments = {}
            if response.data:
                for ack in response.data:
                    doc_name = ack.get("document_name", "")
                    acknowledgments[doc_name] = {
                        "acknowledged": True,
                        "acknowledged_at": ack.get("acknowledged_at")
                    }
            
            return acknowledgments
        except Exception as e:
            print(f"Error fetching user acknowledgments: {e}")
            return {}

    def find_document_acknowledgment(self, pdf_name: str, doc_title: str, user_acknowledgments: dict) -> dict:
        """Find acknowledgment using multiple matching strategies."""
        
        # Strategy 1: Exact match with doc_title (most common)
        if doc_title in user_acknowledgments:
            return user_acknowledgments[doc_title]
        
        # Strategy 2: Exact match with pdf_name
        if pdf_name in user_acknowledgments:
            return user_acknowledgments[pdf_name]
        
        # Strategy 3: Case-insensitive matching
        doc_title_lower = doc_title.lower() if doc_title else ""
        pdf_name_lower = pdf_name.lower() if pdf_name else ""
        
        for ack_key, ack_data in user_acknowledgments.items():
            ack_key_lower = ack_key.lower()
            if doc_title_lower == ack_key_lower or pdf_name_lower == ack_key_lower:
                return ack_data
        
        # Strategy 4: Partial matching
        for ack_key, ack_data in user_acknowledgments.items():
            if (doc_title and doc_title.lower() in ack_key.lower()) or \
               (pdf_name and pdf_name.lower() in ack_key.lower()):
                return ack_data
        
        return {}

    def process_request(self, req: OnboardingFetchRequest, supabase_client, get_signed_url, list_files_in_folder, check_file_exists) -> OnboardingFetchResponse:
        """Process a Onboarding document fetch request."""
        try:
            # Get documents for the role
            documents = self.get_role_documents(req.role, supabase_client)

            if not documents:
                return OnboardingFetchResponse(
                    success=False,
                    message=f"No Onboarding documents found for role: {req.role}",
                    documents=[]
                )

            # Get user acknowledgments
            user_acknowledgments = self.get_user_acknowledgments(req.email, supabase_client)

            # Format documents directly using the provided summaries
            formatted_docs = []

            for doc in documents:
                title = doc.get("doc_title", "Untitled Document")
                folder_path = doc.get("gcs_url", "")
                pdf_name = doc.get("pdf_name", "")                # Construct the file path
                file_path = f"{folder_path}/{pdf_name}" if folder_path and pdf_name else ""

                # Get signed URL only if file path exists
                url = ""
                if file_path and check_file_exists(file_path):
                    url = get_signed_url(file_path)
                
                # Check if the document is acknowledged by the user
                acknowledgment = self.find_document_acknowledgment(pdf_name, title, user_acknowledgments)
                acknowledged = acknowledgment.get("acknowledged", False)
                acknowledged_at = acknowledgment.get("acknowledged_at")
                
                # Create document with acknowledgment status
                formatted_docs.append(Document(
                    title=title,
                    url=url,
                    acknowledged=acknowledged,
                    acknowledged_at=acknowledged_at
                ))

            if not formatted_docs:
                return OnboardingFetchResponse(
                    success=False,
                    message=f"No accessible Onboarding documents found for role: {req.role}",
                    documents=[],
                    acknowledgment_summary=AcknowledgmentSummary(
                        total_documents=0,
                        acknowledged_count=0,
                        pending_count=0
                    )
                )

            # Calculate acknowledgment summary
            total_documents = len(formatted_docs)
            acknowledged_count = sum(1 for doc in formatted_docs if doc.acknowledged)
            pending_count = total_documents - acknowledged_count

            acknowledgment_summary = AcknowledgmentSummary(
                total_documents=total_documents,
                acknowledged_count=acknowledged_count,
                pending_count=pending_count
            )

            return OnboardingFetchResponse(
                success=True,
                message=f"Successfully retrieved {len(formatted_docs)} Onboarding documents",
                documents=formatted_docs,
                acknowledgment_summary=acknowledgment_summary
            )
        except Exception as e:
            print(f"Error processing Onboarding request: {e}")
            return OnboardingFetchResponse(
                success=False,
                message=f"Error processing Onboarding request: {str(e)}",
                documents=[]
            )