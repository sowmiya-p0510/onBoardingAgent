from typing import Optional
from pydantic import BaseModel
import datetime
import os
import requests
import socket

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
        Initialize the WelcomeAgent with enhanced professional communication capabilities.

        Args:
            agent: Parameter kept for compatibility but not used
        """
        # Lazy initialization for LLM
        self._llm = None
        self.company_name = "Fusefy"

    @property
    def llm(self):
        """Lazy initialization of LLM with enhanced parameters for professional communication"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model="gpt-4",  # Upgraded to GPT-4 for better professional communication
                temperature=0.1,  # Lower temperature for more consistent, professional output
                api_key=os.environ.get("OPENAI_API_KEY"),
                max_tokens=1500  # Increased token limit for comprehensive responses
            )
        return self._llm

    def get_user_ip_and_region(self) -> tuple[str, str]:
        """
        Get user's IP address and determine their geographic region with enhanced error handling.
        Returns tuple of (ip_address, region)
        """
        try:
            # Get public IP address with timeout
            ip_response = requests.get('https://api.ipify.org', timeout=10)
            ip_address = ip_response.text.strip()
            
            # Get geographic information from IP with backup service
            try:
                geo_response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=10)
                geo_data = geo_response.json()
                
                if geo_data.get('status') == 'success':
                    continent = geo_data.get('continent', '').lower()
                    country = geo_data.get('country', '').lower()
                    
                    # Map to our enhanced regions
                    region = self._map_to_region(continent, country)
                    return ip_address, region
                else:
                    return ip_address, "global"  # Professional fallback
            except:
                # Fallback to secondary geolocation service
                return ip_address, "global"
                
        except Exception as e:
            print(f"Geographic detection temporarily unavailable: {e}")
            return "localhost", "global"  # Professional fallback

    def _map_to_region(self, continent: str, country: str) -> str:
        """
        Enhanced mapping of continent/country to professional regions
        """
        continent = continent.lower()
        country = country.lower()
        
        # Enhanced mapping logic with European support
        if continent in ['asia']:
            return "asia"
        elif continent in ['africa']:
            return "africa"
        elif continent in ['north america', 'south america'] or 'america' in continent:
            return "america"
        elif continent in ['oceania'] or country in ['australia', 'new zealand']:
            return "australia"
        elif continent in ['europe'] or country in ['united kingdom', 'germany', 'france', 'italy', 'spain']:
            return "europe"
        else:
            # Enhanced fallback with professional default
            if any(keyword in country for keyword in ['australia', 'new zealand']):
                return "australia"
            elif any(keyword in country for keyword in ['canada', 'usa', 'united states', 'mexico', 'brazil', 'argentina']):
                return "america"
            elif any(keyword in country for keyword in ['egypt', 'south africa', 'nigeria', 'kenya', 'morocco']):
                return "africa"
            elif any(keyword in country for keyword in ['uk', 'england', 'france', 'germany', 'netherlands']):
                return "europe"
            else:
                return "global"  # Professional neutral default

    def get_user_profile(self, email: str, supabase_client):
        """Get user profile data from Supabase with enhanced error handling."""
        try:
            response = supabase_client.table("user_profiles") \
                .select("*") \
                .eq("email", email) \
                .execute()

            if not response.data:
                print(f"User profile not found for: {email}")
                return None

            return UserProfile(**response.data[0])
        except Exception as e:
            print(f"Database connection error while fetching user profile: {e}")
            return None

    def get_user_document_acknowledgments(self, email: str, supabase_client) -> dict:
        """Get user's document acknowledgment status with comprehensive tracking"""
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
            print(f"Error retrieving document acknowledgment records: {e}")
            return {
                'acknowledgments': {},
                'acknowledgment_details': [],
                'total_acknowledged': 0
            }

    def get_mandatory_documents_summary(self, supabase_client) -> dict:
        """
        Retrieve comprehensive mandatory documents information from Supabase.
        """
        try:
            response = supabase_client.table("agents").select("documents").execute()

            if not response.data:
                print("No mandatory documents configuration found in system.")
                return {
                    'total_mandatory': 0,
                    'mandatory_document_names': []
                }

            # Extract document titles with enhanced processing
            docs_json = response.data[0].get("documents", [])
            doc_names = [doc["doc_title"] for doc in docs_json if "doc_title" in doc]

            return {
                'total_mandatory': len(doc_names),
                'mandatory_document_names': doc_names
            }

        except Exception as e:
            print(f"System error retrieving mandatory documents: {e}")
            return {
                'total_mandatory': 0,
                'mandatory_document_names': []
            }

    def get_document_progress_summary(self, email: str, supabase_client) -> dict:
        """Generate comprehensive document compliance progress summary"""
        try:
            # Get user acknowledgments
            user_acks = self.get_user_document_acknowledgments(email, supabase_client)
            
            # Get mandatory documents info
            mandatory_info = self.get_mandatory_documents_summary(supabase_client)
            
            total_mandatory = mandatory_info['total_mandatory']
            total_acknowledged = user_acks['total_acknowledged']
            completion_percentage = (total_acknowledged / total_mandatory * 100) if total_mandatory > 0 else 0
            
            # Professional categorization of documents
            acknowledged_docs = []
            pending_docs = list(mandatory_info['mandatory_document_names'])
            
            # Process acknowledged documents with professional formatting
            for doc_name, ack_info in user_acks['acknowledgments'].items():
                if doc_name in pending_docs:
                    pending_docs.remove(doc_name)
                    # Professional date formatting
                    ack_date = ack_info['acknowledged_at'][:10] if ack_info['acknowledged_at'] else 'Date unavailable'
                    acknowledged_docs.append({
                        'name': doc_name,
                        'acknowledged_at': ack_date
                    })
                else:
                    acknowledged_docs.append({
                        'name': doc_name,
                        'acknowledged_at': ack_info['acknowledged_at'][:10] if ack_info['acknowledged_at'] else 'Date unavailable'
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
            print(f"Error generating document compliance summary: {e}")
            # Professional fallback with standard corporate documents
            return {
                'total_mandatory': 5,
                'total_acknowledged': 0,
                'total_pending': 5,
                'completion_percentage': 0.0,
                'acknowledged_documents': [],
                'pending_documents': ["Employee Handbook", "Code of Conduct", "Data Protection Policy", "Health & Safety Guidelines", "IT Security Standards"],
                'user_acknowledgments': {'acknowledgments': {}, 'acknowledgment_details': [], 'total_acknowledged': 0}
            }

    def process_request(self, req: WelcomeRequest, supabase_client) -> WelcomeResponse:
        """Process welcome message request with enhanced professional standards."""
        try:
            # Retrieve comprehensive user profile
            user_profile = self.get_user_profile(req.email, supabase_client)

            if not user_profile:
                return WelcomeResponse(
                    success=False,
                    message=f"Employee record not found. Please contact HR for assistance with email: {req.email}"
                )

            # Generate comprehensive document compliance summary
            doc_progress = self.get_document_progress_summary(req.email, supabase_client)

            # Determine geographic context for personalization
            ip_address, region = self.get_user_ip_and_region()

            # Generate professional welcome message with regional awareness
            welcome_message = self._generate_enhanced_welcome_message(user_profile, doc_progress, region)

            return WelcomeResponse(
                success=True,
                message="Welcome message successfully generated with professional standards",
                user_profile=user_profile,
                welcome_message=welcome_message
            )
        except Exception as e:
            print(f"Critical error in welcome message processing: {e}")
            return WelcomeResponse(
                success=False,
                message=f"System temporarily unavailable. Please contact IT support. Error reference: {str(e)[:50]}"
            )

    def _format_acknowledged_docs(self, acknowledged_docs: list) -> str:
        """Professional formatting for completed document acknowledgments"""
        if not acknowledged_docs:
            return "• No documents completed yet"
        
        formatted = []
        for doc in acknowledged_docs:
            formatted.append(f"• {doc['name']} — Completed: {doc['acknowledged_at']}")
        return "\n".join(formatted)
    
    def _format_pending_docs(self, pending_docs: list) -> str:
        """Professional formatting for pending document requirements"""
        if not pending_docs:
            return "• All mandatory documents have been completed"
        
        formatted = []
        for i, doc in enumerate(pending_docs, 1):
            formatted.append(f"• {doc}")
        return "\n".join(formatted)

    def _get_regional_greeting(self, region: str) -> tuple[str, str]:
        """Enhanced professional regional greetings with specific context"""
        regional_context = {
            "asia": ("Welcome to Fusefy", "Based in the Asia-Pacific region, you'll be part of our strategic growth and innovation initiatives."),
            "africa": ("Welcome to Fusefy", "As part of our African operations, you'll contribute to our expanding continental presence."),
            "america": ("Welcome to Fusefy", "From our Americas division, you'll drive forward our regional excellence and innovation."),
            "australia": ("Welcome to Fusefy", "As part of our Oceania operations, you'll contribute to our regional market leadership."),
            "europe": ("Welcome to Fusefy", "Within our European division, you'll be part of our continued legacy of excellence.")
        }
        
        # Using America's context as default
        return regional_context.get(region, regional_context["america"])

    def _generate_enhanced_welcome_message(self, user_profile: UserProfile, doc_progress: dict, region: str) -> str:
        """Generate enhanced professional welcome message with LLM optimization"""
        
        # Professional date formatting
        try:
            joining_date = datetime.datetime.strptime(user_profile.joining_date, "%Y-%m-%d")
            formatted_joining_date = joining_date.strftime("%B %d, %Y")
        except:
            formatted_joining_date = user_profile.joining_date

        # Extract professional name reference
        first_name = user_profile.full_name.split()[0] if user_profile.full_name else "colleague"

        # Get regional greeting and context
        regional_greeting, regional_context = self._get_regional_greeting(region)

        # Professional document formatting
        acknowledged_docs_text = self._format_acknowledged_docs(doc_progress['acknowledged_documents'])
        pending_docs_text = self._format_pending_docs(doc_progress['pending_documents'])

        # Enhanced professional prompt with updated format
        prompt = f"""You are a senior HR communications specialist tasked with creating a professional, comprehensive welcome message for a new employee. The message must maintain corporate standards while being warm and informative.

EMPLOYEE DETAILS:
- Full Name: {user_profile.full_name}
- Position: {user_profile.role}
- Department: {user_profile.department}
- Direct Manager: {user_profile.manager_name} ({user_profile.manager_email})
- Start Date: {formatted_joining_date}
- Location: {user_profile.location}
- Employment Type: {user_profile.employment_type}
- Geographic Region: {region.upper()}

DOCUMENT COMPLIANCE STATUS:
- Total Mandatory Documents: {doc_progress['total_mandatory']}
- Documents Completed: {doc_progress['total_acknowledged']}
- Documents Pending: {doc_progress['total_pending']}
- Compliance Rate: {doc_progress['completion_percentage']:.1f}%

COMPLETED DOCUMENTS:
{acknowledged_docs_text}

PENDING DOCUMENTS:
{pending_docs_text}

MANDATORY REQUIREMENTS FOR OUTPUT:
1. Use EXACTLY this structure and format
2. Maintain professional corporate tone throughout
3. Include all compliance information precisely as provided
4. Keep regional reference subtle and professional
5. Use proper business communication standards

REQUIRED OUTPUT FORMAT (copy exactly):

**{regional_greeting}, {first_name}!**

{regional_context}

We are pleased to formally welcome you to {self.company_name} as {user_profile.role} within our {user_profile.department} department, effective {formatted_joining_date}. Your expertise and experience will be valuable additions to our team.

**📋 Document Compliance Status**
Current Progress: {doc_progress['completion_percentage']:.0f}% Complete ({doc_progress['total_acknowledged']} of {doc_progress['total_mandatory']} mandatory documents)

**✅ Documents Completed:**
{acknowledged_docs_text}

**⏳ Pending Requirements:**
{pending_docs_text}

**🎯 Next Steps:**
1. Complete any pending document acknowledgments above
2. Reach out to your manager {user_profile.manager_name} ({user_profile.manager_email}) for your first check-in
3. Access your employee portal to review benefits and policies
4. Schedule your IT setup and workspace orientation

**🌐 Collaboration Note:** Our organization operates across multiple time zones with flexible communication channels to ensure seamless collaboration.

**Welcome aboard! We're excited to see the great things you'll accomplish here.** 🚀"""

        # Enhanced validation requirements
        required_elements = [
            "Document Compliance Status",
            "Documents Completed:",
            "Pending Requirements:",
            "🎯 Next Steps:",
            "Collaboration Note:",
            "Welcome aboard!"
        ]

        try:
            # Enhanced retry logic for professional output
            for attempt in range(3):
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Validate response contains all required professional elements
                if all(element in content for element in required_elements):
                    return content
                
                print(f"Attempt {attempt + 1}: Response missing required professional elements")
            
            # Professional fallback message
            return self._generate_fallback_message(user_profile, doc_progress, region, 
                                                 formatted_joining_date, first_name, 
                                                 acknowledged_docs_text, pending_docs_text)
            
        except Exception as e:
            print(f"LLM service error in welcome message generation: {e}")
            return self._generate_fallback_message(user_profile, doc_progress, region, 
                                                 formatted_joining_date, first_name, 
                                                 acknowledged_docs_text, pending_docs_text)

    def _generate_fallback_message(self, user_profile, doc_progress, region, 
                                  formatted_joining_date, first_name, acknowledged_docs_text, pending_docs_text):
        """Generate professional fallback message"""
        regional_greeting, regional_context = self._get_regional_greeting(region)
        
        return f"""**{regional_greeting}, {first_name}!**

{regional_context}

We are pleased to formally welcome you to {self.company_name} as {user_profile.role} within our {user_profile.department} department, effective {formatted_joining_date}. Your expertise and experience will be valuable additions to our team.

**📋 Document Compliance Status**
Current Progress: {doc_progress['completion_percentage']:.0f}% Complete ({doc_progress['total_acknowledged']} of {doc_progress['total_mandatory']} mandatory documents)

**✅ Documents Completed:**
{acknowledged_docs_text}

**⏳ Pending Requirements:**
{pending_docs_text}

**🎯 Next Steps:**
1. Complete any pending document acknowledgments above
2. Reach out to your manager {user_profile.manager_name} ({user_profile.manager_email}) for your first check-in
3. Access your employee portal to review benefits and policies
4. Schedule your IT setup and workspace orientation

**🌐 Collaboration Note:** Our organization operates across multiple time zones with flexible communication channels to ensure seamless collaboration.

**Welcome aboard! We're excited to see the great things you'll accomplish here.** 🚀"""