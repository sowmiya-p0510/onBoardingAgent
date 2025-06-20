from typing import Optional
from pydantic import BaseModel
import datetime
import os

# --- Models ---

class WelcomeRequest(BaseModel):
    email: str

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: str
    department: str
    manager_name: str
    manager_email: str
    joining_date: str
    location: str
    employment_type: str
    created_at: str

class WelcomeResponse(BaseModel):
    success: bool
    message: str
    user_profile: Optional[UserProfile] = None
    welcome_message: Optional[str] = None

# --- Welcome Agent ---

class WelcomeAgent:
    def __init__(self, agent=None):
        """
        Initialize the WelcomeAgent.

        Args:
            agent: Parameter kept for compatibility but not used
        """
        # Lazy initialization for LLM
        self._llm = None
        self.company_name = "Fusefy"

    @property
    def llm(self):
        """Lazy initialization of LLM"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        return self._llm

    def get_user_profile(self, email: str, supabase_client):
        """Get user profile data from Supabase."""
        try:
            response = supabase_client.table("user_profiles") \
                .select("*") \
                .eq("email", email) \
                .execute()

            if not response.data:
                return None

            return UserProfile(**response.data[0])
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            return None

    def get_user_document_acknowledgments(self, email: str, supabase_client) -> dict:
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

    def get_mandatory_documents_summary(self, supabase_client) -> dict:
        """
        Dynamically fetch actual mandatory documents from Supabase (agents table).
        """
        try:
            response = supabase_client.table("agents").select("documents").execute()

            if not response.data:
                print("No agent documents found in Supabase.")
                return {
                    'total_mandatory': 0,
                    'mandatory_document_names': []
                }

            # Flatten all document titles from all agents (or pick one specific agent if needed)
            docs_json = response.data[0].get("documents", [])
            doc_names = [doc["doc_title"] for doc in docs_json if "doc_title" in doc]

            return {
                'total_mandatory': len(doc_names),
                'mandatory_document_names': doc_names
            }

        except Exception as e:
            print(f"Error fetching documents from Supabase: {e}")
            return {
                'total_mandatory': 0,
                'mandatory_document_names': []
            }


    def get_document_progress_summary(self, email: str, supabase_client) -> dict:
        """Get comprehensive document progress summary"""
        try:
            # Get user acknowledgments
            user_acks = self.get_user_document_acknowledgments(email, supabase_client)
            
            # Get mandatory documents info
            mandatory_info = self.get_mandatory_documents_summary(supabase_client)
            
            total_mandatory = mandatory_info['total_mandatory']
            total_acknowledged = user_acks['total_acknowledged']
            completion_percentage = (total_acknowledged / total_mandatory * 100) if total_mandatory > 0 else 0
            
            # Categorize acknowledged vs pending
            acknowledged_docs = []
            pending_docs = list(mandatory_info['mandatory_document_names'])  # Start with all as pending
            
            # Remove acknowledged docs from pending and add to acknowledged
            for doc_name, ack_info in user_acks['acknowledgments'].items():
                if doc_name in pending_docs:
                    pending_docs.remove(doc_name)
                    # Format date properly for display
                    ack_date = ack_info['acknowledged_at'][:10] if ack_info['acknowledged_at'] else 'Unknown'
                    acknowledged_docs.append({
                        'name': doc_name,
                        'acknowledged_at': ack_date
                    })
                else:
                    # Document was acknowledged but might not be in mandatory list
                    acknowledged_docs.append({
                        'name': doc_name,
                        'acknowledged_at': ack_info['acknowledged_at'][:10] if ack_info['acknowledged_at'] else 'Unknown'
                    })
            
            return {
                'total_mandatory': total_mandatory,
                'total_acknowledged': total_acknowledged,
                'total_pending': len(pending_docs),
                'completion_percentage': round(completion_percentage, 1),
                'acknowledged_documents': acknowledged_docs,
                'pending_documents': pending_docs,
                'user_acknowledgments': user_acks
            }
        except Exception as e:
            print(f"Error getting document progress: {e}")
            return {
                'total_mandatory': 5,
                'total_acknowledged': 0,
                'total_pending': 5,
                'completion_percentage': 0.0,
                'acknowledged_documents': [],
                'pending_documents': ["Employee Handbook", "Code of Conduct", "Safety Guidelines", "IT Security Policy", "Benefits Overview"],
                'user_acknowledgments': {'acknowledgments': {}, 'acknowledgment_details': [], 'total_acknowledged': 0}
            }

    def process_request(self, req: WelcomeRequest, supabase_client) -> WelcomeResponse:
        """Process a welcome message request with LLM generation."""
        try:
            # Get user profile data
            user_profile = self.get_user_profile(req.email, supabase_client)

            if not user_profile:
                return WelcomeResponse(
                    success=False,
                    message=f"No user profile found for email: {req.email}"
                )

            # Get document progress summary
            doc_progress = self.get_document_progress_summary(req.email, supabase_client)

            # Generate LLM-powered welcome message with progression tracking
            welcome_message = self._generate_llm_welcome_message(user_profile, doc_progress)

            return WelcomeResponse(
                success=True,
                message="Successfully generated welcome message",
                user_profile=user_profile,
                welcome_message=welcome_message
            )
        except Exception as e:
            print(f"Error processing welcome request: {e}")
            return WelcomeResponse(
                success=False,
                message=f"Error processing welcome request: {str(e)}"
            )

    def _format_acknowledged_docs(self, acknowledged_docs: list) -> str:
        """Format acknowledged documents list for display"""
        if not acknowledged_docs:
            return "• None yet"
        
        formatted = []
        for doc in acknowledged_docs:
            formatted.append(f"• {doc['name']} (completed {doc['acknowledged_at']})")
        return "\n".join(formatted)
    
    def _format_pending_docs(self, pending_docs: list) -> str:
        """Format pending documents list for display"""
        if not pending_docs:
            return "• All documents completed!"
        
        formatted = []
        for doc in pending_docs:
            formatted.append(f"• {doc}")
        return "\n".join(formatted)

    def _generate_llm_welcome_message(self, user_profile: UserProfile, doc_progress: dict) -> str:
        """Generate a personalized welcome message using LLM with document progression tracking."""
        
        # Format joining date for better readability
        try:
            joining_date = datetime.datetime.strptime(user_profile.joining_date, "%Y-%m-%d")
            formatted_joining_date = joining_date.strftime("%B %d, %Y")
        except:
            formatted_joining_date = user_profile.joining_date

        # Extract first name for personalization
        first_name = user_profile.full_name.split()[0] if user_profile.full_name else "there"

        # Create user context for LLM
        user_context = f"""
Employee Profile:
- Name: {user_profile.full_name}
- Role: {user_profile.role}
- Department: {user_profile.department}
- Manager: {user_profile.manager_name} ({user_profile.manager_email})
- Start Date: {formatted_joining_date}
- Location: {user_profile.location}
- Employment Type: {user_profile.employment_type}
"""

        # Create document progress context
        progress_context = f"""
Document Acknowledgment Progress:
- Total Mandatory Documents: {doc_progress['total_mandatory']}
- Documents Acknowledged: {doc_progress['total_acknowledged']}
- Documents Pending: {doc_progress['total_pending']}
- Completion Percentage: {doc_progress['completion_percentage']:.1f}%
"""

        # Add acknowledged documents details
        if doc_progress['acknowledged_documents']:
            progress_context += "\nCompleted Acknowledgments:\n"
            for doc in doc_progress['acknowledged_documents']:
                progress_context += f"- {doc['name']} (completed {doc['acknowledged_at']})\n"

        # Add pending documents
        if doc_progress['pending_documents']:
            progress_context += "\nPending Acknowledgments:\n"
            for doc in doc_progress['pending_documents']:
                progress_context += f"- {doc}\n"        # Format document lists for the prompt
        acknowledged_docs_text = self._format_acknowledged_docs(doc_progress['acknowledged_documents'])
        pending_docs_text = self._format_pending_docs(doc_progress['pending_documents'])

        # Create LLM prompt with structured format
        prompt = f"""Generate a professional welcome message for a new employee at {self.company_name}. Follow this EXACT structure and format:

{user_context}

{progress_context}

REQUIRED OUTPUT FORMAT (copy exactly, filling in the data):

**Welcome to {self.company_name}, {first_name}! 🎉**
We're thrilled to have you join us as {user_profile.role} in the {user_profile.department} department, starting {formatted_joining_date}.

**📋 Document Acknowledgment Progress**
Progress: {doc_progress['completion_percentage']:.0f}% Complete ({doc_progress['total_acknowledged']}/{doc_progress['total_mandatory']} documents)

✅ **Completed:**
{acknowledged_docs_text}

⏳ **Pending:**
{pending_docs_text}

**🎯 Next Steps:**
1. Complete any pending document acknowledgments above
2. Reach out to your manager {user_profile.manager_name} ({user_profile.manager_email}) for your first check-in
3. Access your employee portal to review benefits and policies
4. Schedule your IT setup and workspace orientation

**Welcome aboard! We're excited to see the great things you'll accomplish here.** 🚀

---
*If you have any questions, don't hesitate to reach out to your manager or HR team.*

CRITICAL: Output ONLY the welcome message content starting with "**Welcome to {self.company_name}..." - do not include any other text, explanations, or formatting instructions in your response."""

        try:
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating LLM welcome message: {e}")
            raise e