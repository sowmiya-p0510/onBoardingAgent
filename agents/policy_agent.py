from typing import List, Optional
from pydantic import BaseModel

class PolicyFetchRequest(BaseModel):
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

class PolicyFetchResponse(BaseModel):
    success: bool
    message: str
    documents: List[Document] = []
    acknowledgment_summary: Optional[AcknowledgmentSummary] = None

class PolicyAgent:
    def __init__(self, agent):
        """
        Initialize the PolicyAgent with a pre-defined agent.

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
                .eq("agent_type", "policy") \
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
            
            # Create a dictionary for quick lookup
            acknowledgments = {}
            if response.data:
                for ack in response.data:
                    doc_name = ack.get("document_name", "")
                    # Remove .pdf extension for matching
                    doc_key = doc_name.replace(".pdf", "")
                    acknowledgments[doc_key] = {
                        "acknowledged": True,
                        "acknowledged_at": ack.get("acknowledged_at")
                    }
            
            return acknowledgments
        except Exception as e:
            print(f"Error fetching user acknowledgments: {e}")
            return {}

    def process_request(self, req: PolicyFetchRequest, supabase_client, get_signed_url, list_files_in_folder, check_file_exists) -> PolicyFetchResponse:
        """Process a policy document fetch request."""
        try:
            # Get documents for the role
            documents = self.get_role_documents(req.role, supabase_client)

            if not documents:
                return PolicyFetchResponse(
                    success=False,
                    message=f"No policy documents found for role: {req.role}",
                    documents=[]
                )

            # Get user acknowledgments
            user_acknowledgments = self.get_user_acknowledgments(req.email, supabase_client)

            # Format documents directly using the provided summaries
            formatted_docs = []

            for doc in documents:
                title = doc.get("doc_title", "Untitled Document")
                folder_path = doc.get("gcs_url", "")
                pdf_name = doc.get("pdf_name", "")

                # Construct the file path
                file_path = f"{folder_path}/{pdf_name}" if folder_path and pdf_name else ""

                # Get signed URL only if file path exists
                url = ""
                if file_path and check_file_exists(file_path):
                    url = get_signed_url(file_path)

                # Check if the document is acknowledged by the user
                doc_key = pdf_name.replace(".pdf", "")  # Match the key used in acknowledgments
                acknowledgment = user_acknowledgments.get(doc_key, {})
                acknowledged = acknowledgment.get("acknowledged", False)
                acknowledged_at = acknowledgment.get("acknowledged_at")                # Create document with acknowledgment status
                formatted_docs.append(Document(
                    title=title,
                    url=url,
                    acknowledged=acknowledged,
                    acknowledged_at=acknowledged_at
                ))

            if not formatted_docs:
                return PolicyFetchResponse(
                    success=False,
                    message=f"No accessible policy documents found for role: {req.role}",
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

            return PolicyFetchResponse(
                success=True,
                message=f"Successfully retrieved {len(formatted_docs)} policy documents",
                documents=formatted_docs,
                acknowledgment_summary=acknowledgment_summary
            )
        except Exception as e:
            print(f"Error processing policy request: {e}")
            return PolicyFetchResponse(
                success=False,
                message=f"Error processing policy request: {str(e)}",
                documents=[]
            )