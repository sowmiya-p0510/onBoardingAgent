from typing import List, Optional
from pydantic import BaseModel

class BenefitFetchRequest(BaseModel):
    role: str
    email: str

class Document(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
    summary: Optional[str] = None

class BenefitFetchResponse(BaseModel):
    success: bool
    message: str
    documents: List[Document] = []

class BenefitAgent:
    def __init__(self, agent):
        """
        Initialize the BenefitAgent with a pre-defined agent.

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
                .eq("agent_type", "benefit") \
                .execute()

            if not response.data:
                return []

            agent_data = response.data[0]
            return agent_data.get("documents", [])
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def process_request(self, req: BenefitFetchRequest, supabase_client, get_signed_url, list_files_in_folder, check_file_exists) -> BenefitFetchResponse:
        """Process a benefit document fetch request."""
        try:
            # Get documents for the role
            documents = self.get_role_documents(req.role, supabase_client)

            if not documents:
                return BenefitFetchResponse(
                    success=False,
                    message=f"No benefit documents found for role: {req.role}",
                    documents=[]
                )

            # Format documents directly using the provided summaries
            formatted_docs = []

            for doc in documents:
                title = doc.get("doc_title", "Untitled Document")
                folder_path = doc.get("gcs_url", "")
                description = doc.get("description", "")
                summary = doc.get("summary", "")
                pdf_name = doc.get("pdf_name", "")

                # Construct the file path
                file_path = f"{folder_path}/{pdf_name}" if folder_path and pdf_name else ""

                # Get signed URL only if file path exists
                url = ""
                if file_path and check_file_exists(file_path):
                    url = get_signed_url(file_path)

                # Create document with existing summary
                formatted_docs.append(Document(
                    title=title,
                    url=url,
                    description=description,
                    summary=summary
                ))

            if not formatted_docs:
                return BenefitFetchResponse(
                    success=False,
                    message=f"No accessible benefit documents found for role: {req.role}",
                    documents=[]
                )

            return BenefitFetchResponse(
                success=True,
                message=f"Successfully retrieved {len(formatted_docs)} benefit documents",
                documents=formatted_docs
            )
        except Exception as e:
            print(f"Error processing benefit request: {e}")
            return BenefitFetchResponse(
                success=False,
                message=f"Error processing benefit request: {str(e)}",
                documents=[]
            )