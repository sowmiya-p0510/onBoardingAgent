from typing import List, Optional
from pydantic import BaseModel
from crewai import Task, Crew
import requests
import os

# --- Models ---

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
    overall_summary: Optional[str] = None

# --- Benefit Agent ---

class BenefitAgent:
    def __init__(self, agent):
        """
        Initialize the BenefitAgent with a pre-defined agent.

        Args:
            agent: A CrewAI Agent instance to use for processing
        """
        self.agent = agent

    def extract_text_from_document(self, url: str) -> str:
        """
        Extract text from a document URL, handling different file types.
        """
        try:
            # Download the file content
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # For now, just return the first part of the response as text
            # In a real implementation, you'd want to use PyMuPDF or similar libraries
            # to properly extract text from PDFs and other document formats
            return response.text[:5000]  # First 5000 characters as a sample
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return ""

    # In agents/benefit_agent.py
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
            documents = agent_data.get("documents", [])

            # Process documents to add public URLs
            processed_docs = []
            for doc in documents:
                if isinstance(doc, str):
                    doc = {
                        "doc_title": os.path.basename(doc).split(".")[0].replace("_", " ").title(),
                        "gcs_url": doc,
                        "description": ""
                    }

                processed_docs.append(doc)

            return processed_docs
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
                    documents=[],
                    overall_summary="No benefit information available for your role."
                )

            # Format documents for the agent
            formatted_docs = []
            doc_summaries = []

            for doc in documents:
                title = doc.get("doc_title", "Untitled Document")
                folder_path = doc.get("gcs_url", "")
                description = doc.get("description", "")
                pdf_name = doc.get("pdf_name", "")

                # If we have a specific PDF name, use it directly
                if pdf_name:
                    file_path = f"{folder_path}/{pdf_name}" if folder_path else pdf_name

                    # Check if file exists
                    if check_file_exists(file_path):
                        url = get_signed_url(file_path)
                        if url:
                            # Create document and process it
                            formatted_doc = self._process_document(title, url, description, req.role)
                            formatted_docs.append(formatted_doc)
                            doc_summaries.append(f"Document: {title}\nSummary: {formatted_doc.summary}")
                        else:
                            print(f"Warning: Could not generate URL for {file_path}")
                    else:
                        print(f"Warning: File does not exist: {file_path}")
                else:
                    # List all files in the folder and process each one
                    files = list_files_in_folder(folder_path)
                    if not files:
                        print(f"Warning: No files found in folder {folder_path}")

                    for file_path in files:
                        file_title = os.path.basename(file_path).split(".")[0].replace("_", " ").title()
                        url = get_signed_url(file_path)
                        if url:
                            # Create document and process it
                            formatted_doc = self._process_document(file_title, url, description, req.role)
                            formatted_docs.append(formatted_doc)
                            doc_summaries.append(f"Document: {file_title}\nSummary: {formatted_doc.summary}")

            if not formatted_docs:
                return BenefitFetchResponse(
                    success=False,
                    message=f"No accessible benefit documents found for role: {req.role}",
                    documents=[],
                    overall_summary="No benefit information available for your role."
                )

            # Generate overall summary
            overall_summary = self._generate_overall_summary(doc_summaries, req.role)

            return BenefitFetchResponse(
                success=True,
                message=f"Successfully retrieved {len(formatted_docs)} benefit documents",
                documents=formatted_docs,
                overall_summary=overall_summary
            )
        except Exception as e:
            print(f"Error processing benefit request: {e}")
            return BenefitFetchResponse(
                success=False,
                message=f"Error processing benefit request: {str(e)}",
                documents=[],
                overall_summary="Unable to process benefits information at this time."
            )
    
    def _process_document(self, title, url, description, role):
        """Process a single document and generate its summary."""
        summary_task = Task(
            description=f"""
            Please analyze and summarize the following benefit document:

            DOCUMENT TITLE: {title}
            DOCUMENT DESCRIPTION: {description}
            DOCUMENT URL: {url}

            Your task is to create a clear, concise summary that helps an employee in the {role} role 
            understand this specific benefit. Focus on:

            1. The core purpose of this benefit
            2. Key eligibility requirements or conditions
            3. The most important details an employee should know
            4. Any deadlines or time-sensitive information
            5. How this benefit provides value to the employee

            Your summary should be 2-3 sentences long, easy to understand, and highlight the most 
            relevant aspects for a {role}. Avoid jargon and technical terms unless absolutely necessary.

            If you cannot access the document content, base your summary on the title and description,
            focusing on what would likely be included in such a document.
            """,
            expected_output="A clear, concise 2-3 sentence summary that captures the essential information about this benefit",
            agent=self.agent
        )

        # For simplicity, use the agent to execute the task directly
        summary_crew = Crew(
            agents=[self.agent],
            tasks=[summary_task],
            verbose=False
        )

        try:
            crew_result = summary_crew.kickoff()
            # Extract the string content from the CrewOutput object
            if hasattr(crew_result, 'raw'):
                summary = str(crew_result.raw)
            else:
                summary = str(crew_result)
        except Exception as e:
            print(f"Error generating summary for {title}: {e}")
            summary = f"Summary unavailable for {title}. Please refer to the document directly."

        return Document(
            title=title,
            url=url,
            description=description,
            summary=summary
        )

    def _generate_overall_summary(self, doc_summaries, role):
        """Generate an overall summary from individual document summaries."""
        overall_task = Task(
            description=f"""
            As a benefits specialist, create a comprehensive overview of the complete benefits package 
            for an employee in the {role} role based on the following document summaries:

            {"\n\n".join(doc_summaries)}

            Your task is to synthesize this information into a cohesive, well-organized summary that helps 
            the employee understand their complete benefits package. Your overview should:

            1. Start with a brief introduction to the overall benefits philosophy of the company
            2. Group related benefits together (healthcare, retirement, time off, etc.)
            3. Highlight the most valuable or unique benefits
            4. Explain how the benefits work together to provide comprehensive coverage
            5. Note any role-specific benefits that are particularly relevant to a {role}
            6. Include any important deadlines, enrollment periods, or action items
            7. End with a brief conclusion about the value of the complete package

            Make your overview clear, informative, and actionable. Use a friendly, supportive tone that 
            helps the employee feel valued and understand how to make the most of their benefits.
            """,
            expected_output="A comprehensive, well-structured overview of the complete benefits package",
            agent=self.agent
        )

        # Create the Crew for overall summary
        crew = Crew(
            agents=[self.agent],
            tasks=[overall_task],
            verbose=True
        )

        # Run the crew and convert output to string
        crew_result = crew.kickoff()
        if hasattr(crew_result, 'raw'):
            overall_summary = str(crew_result.raw)
        else:
            overall_summary = str(crew_result)

        return overall_summary