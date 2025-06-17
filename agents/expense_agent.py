from typing import List, Optional
from pydantic import BaseModel
from crewai import Task, Crew
import requests
import os

# --- Models ---

class ExpenseFetchRequest(BaseModel):
    role: str
    email: str

class Document(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
    summary: Optional[str] = None

class ExpenseFetchResponse(BaseModel):
    success: bool
    message: str
    documents: List[Document] = []
    overall_summary: Optional[str] = None

# --- Expense Agent ---

class ExpenseAgent:
    def __init__(self, agent):
        """
        Initialize the ExpenseAgent with a pre-defined agent.

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

    def get_role_documents(self, role: str, supabase_client):
        """Get documents for a specific role from Supabase."""
        try:
            response = supabase_client.table("agents") \
                .select("*") \
                .eq("role", role) \
                .eq("agent_type", "expense") \
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

    def process_request(self, req: ExpenseFetchRequest, supabase_client, get_signed_url, list_files_in_folder, check_file_exists) -> ExpenseFetchResponse:
        """Process an expense document fetch request."""
        try:
            # Get documents for the role
            documents = self.get_role_documents(req.role, supabase_client)

            if not documents:
                return ExpenseFetchResponse(
                    success=False,
                    message=f"No expense documents found for role: {req.role}",
                    documents=[],
                    overall_summary="No expense information available for your role."
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
                return ExpenseFetchResponse(
                    success=False,
                    message=f"No accessible expense documents found for role: {req.role}",
                    documents=[],
                    overall_summary="No expense information available for your role."
                )

            # Generate overall summary
            overall_summary = self._generate_overall_summary(doc_summaries, req.role)

            return ExpenseFetchResponse(
                success=True,
                message=f"Successfully retrieved {len(formatted_docs)} expense documents",
                documents=formatted_docs,
                overall_summary=overall_summary
            )
        except Exception as e:
            print(f"Error processing expense request: {e}")
            return ExpenseFetchResponse(
                success=False,
                message=f"Error processing expense request: {str(e)}",
                documents=[],
                overall_summary="Unable to process expense information at this time."
            )

    def _process_document(self, title, url, description, role):
        """Process a single document and generate its summary."""
        summary_task = Task(
            description=f"""
            Please analyze and summarize the following expense document:

            DOCUMENT TITLE: {title}
            DOCUMENT DESCRIPTION: {description}
            DOCUMENT URL: {url}

            Your task is to create a clear, concise summary that helps an employee in the {role} role 
            understand this specific expense policy or procedure. Focus on:

            1. The core purpose of this expense document
            2. Key expense submission requirements or conditions
            3. Approval processes and reimbursement timelines
            4. Expense limits or thresholds that apply
            5. Required documentation or receipts
            6. Any deadlines or time-sensitive information
            7. Special considerations for the {role} position

            Your summary should be 2-3 sentences long, easy to understand, and highlight the most 
            relevant aspects for a {role}. Avoid jargon and technical terms unless absolutely necessary.

            If you cannot access the document content, base your summary on the title and description,
            focusing on what would likely be included in such a document.
            """,
            expected_output="A clear, concise 2-3 sentence summary that captures the essential information about this expense document",
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
            As an expense management specialist, create a comprehensive overview of the expense policies 
            and procedures for an employee in the {role} role based on the following document summaries:

            {"\n\n".join(doc_summaries)}

            Your task is to synthesize this information into a cohesive, well-organized summary that helps 
            the employee understand the complete expense management process. Your overview should:

            1. Start with a brief introduction to the company's expense philosophy and expectations
            2. Explain the end-to-end expense submission and reimbursement process
            3. Group related expense policies together (travel, meals, equipment, etc.)
            4. Highlight the most important expense limits, requirements, and deadlines
            5. Explain documentation requirements and approval workflows
            6. Note any role-specific expense considerations that are particularly relevant to a {role}
            7. Include any important tools, systems, or contacts for expense management
            8. End with best practices for efficient expense management and reimbursement

            Make your overview clear, informative, and actionable. Use a friendly, supportive tone that 
            helps the employee understand how to properly manage and submit expenses.
            """,
            expected_output="A comprehensive, well-structured overview of the complete expense management process",
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