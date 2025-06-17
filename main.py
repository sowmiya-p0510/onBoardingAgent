from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware

# Import our agents
from agents.benefit_agent import BenefitAgent, BenefitFetchRequest, BenefitFetchResponse
from agents.policy_agent import PolicyAgent, PolicyFetchRequest, PolicyFetchResponse
# Import our new WelcomeAgent instead of SimpleWelcomeAgent
from agents.welcome_agent import WelcomeAgent, WelcomeRequest, WelcomeResponse, UserProfile

# Import GCS utilities
from utils.gcs_utils import GCSManager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Onboarding System")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for input
class OnboardingRequest(BaseModel):
    name: str
    role: str
    start_date: str
    email: str
    team: str
    manager: str

class OnboardingResponse(BaseModel):
    success: bool
    welcome_message: str
    benefits_summary: str = None
    policy_summary: str = None
    next_steps: list
    documents: list
    team_contacts: dict

# Create the LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

# Define agents with detailed prompts
hr_agent = Agent(
    role='HR Onboarding Specialist',
    goal='Create personalized, comprehensive onboarding materials that make new employees feel welcome and prepared',
    backstory="""You are an experienced HR professional with expertise in employee onboarding.
    You understand that starting a new job can be overwhelming, so you create warm, 
    informative onboarding materials that ease the transition. You have a knack for 
    balancing professional information with a friendly, welcoming tone. You're known 
    for your attention to detail and ability to customize materials for different roles.""",
    llm=llm,
    verbose=True
)

benefit_specialist = Agent(
    role='Employee Benefits Expert',
    goal='Help employees fully understand and maximize their benefits package',
    backstory="""You are a seasoned benefits specialist with deep knowledge of corporate 
    benefit programs. You have a talent for explaining complex benefits information in 
    simple, actionable terms. You understand that employees often find benefits confusing, 
    so you focus on clarity and practical advice. You're particularly skilled at highlighting 
    the most valuable aspects of each benefit and explaining how they work together to 
    provide comprehensive coverage. You always emphasize key deadlines, eligibility 
    requirements, and enrollment procedures.""",
    llm=llm,
    verbose=True
)

policy_specialist = Agent(
    role='Company Policy Expert',
    goal='Help employees understand and comply with company policies',
    backstory="""You are a knowledgeable policy expert with extensive experience in corporate 
    governance and compliance. You excel at explaining complex policies in clear, accessible 
    language. You understand that employees need to know both the letter and spirit of company 
    policies, so you focus on practical explanations and real-world applications. You're skilled 
    at highlighting the most important aspects of each policy and explaining why they matter. 
    You always emphasize key compliance requirements, reporting procedures, and employee 
    responsibilities in a supportive, non-threatening way.""",
    llm=llm,
    verbose=True
)

# Initialize helpers
gcs_manager = GCSManager()

def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)

# Initialize agents
welcome_agent = WelcomeAgent(agent=hr_agent)  # Using our new WelcomeAgent
benefit_agent = BenefitAgent(agent=benefit_specialist)
policy_agent = PolicyAgent(agent=policy_specialist)

@app.post("/onboard", response_model=OnboardingResponse)
async def onboard_employee(req: OnboardingRequest):
    """
    Complete onboarding process using multiple agents

    This endpoint orchestrates agents to provide a comprehensive onboarding experience:
    1. Welcome Agent: Creates personalized welcome message and guidance
    2. Benefits Agent: Fetches and summarizes benefit documents
    3. Policy Agent: Fetches and summarizes policy documents
    """
    try:
        # Get detailed welcome info from Welcome Agent
        supabase = get_supabase_client()
        welcome_request = WelcomeRequest(email=req.email)
        welcome_response = welcome_agent.process_request(welcome_request, supabase_client=supabase)

        # Get benefits information
        benefits_summary = "Benefits information will be provided by HR during your first week."
        try:
            benefit_request = BenefitFetchRequest(role=req.role, email=req.email)
            benefits_response = benefit_agent.process_request(
                benefit_request,
                supabase_client=supabase,
                get_signed_url=gcs_manager.get_signed_url,
                list_files_in_folder=gcs_manager.list_files_in_folder,
                check_file_exists=gcs_manager.check_file_exists
            )
            if benefits_response.success:
                benefits_summary = benefits_response.overall_summary
        except Exception as e:
            print(f"Benefits processing error: {e}")

        # Get policy information
        policy_summary = "Company policies will be provided during your orientation."
        try:
            policy_request = PolicyFetchRequest(role=req.role, email=req.email)
            policy_response = policy_agent.process_request(
                policy_request,
                supabase_client=supabase,
                get_signed_url=gcs_manager.get_signed_url,
                list_files_in_folder=gcs_manager.list_files_in_folder,
                check_file_exists=gcs_manager.check_file_exists
            )
            if policy_response.success:
                policy_summary = policy_response.overall_summary
        except Exception as e:
            print(f"Policy processing error: {e}")

        # Create default response if welcome agent fails
        default_welcome = f"Welcome {req.name}! We're excited to have you join our {req.team} team as our new {req.role}."

        # Prepare response using welcome agent data or defaults
        if welcome_response.success and welcome_response.welcome_message:
            welcome_message = welcome_response.welcome_message
        else:
            welcome_message = default_welcome

        # Extract user profile if available
        user_profile = welcome_response.user_profile if welcome_response.success else None

        # Create next steps and team contacts
        next_steps = [
            f"Contact your manager {user_profile.manager_name if user_profile else req.manager} for guidance",
            "Complete employee documentation",
            "Review company policies"
        ]

        team_contacts = {
            "manager": user_profile.manager_name if user_profile else req.manager,
            "manager_email": user_profile.manager_email if user_profile else "",
            "department": user_profile.department if user_profile else req.team
        }

        return OnboardingResponse(
            success=True,
            welcome_message=welcome_message,
            benefits_summary=benefits_summary,
            policy_summary=policy_summary,
            next_steps=next_steps,
            documents=["Employee Handbook", "Company Policies"],
            team_contacts=team_contacts
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing onboarding: {str(e)}")

@app.post("/welcome/fetch", response_model=WelcomeResponse)
async def generate_welcome(req: WelcomeRequest):
    """Generate personalized welcome message using user profile from database."""
    try:
        supabase = get_supabase_client()
        response = welcome_agent.process_request(req, supabase_client=supabase)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating welcome message: {str(e)}")

@app.post("/benefit/fetch", response_model=BenefitFetchResponse)
async def fetch_benefits(req: BenefitFetchRequest):
    """Fetch and summarize benefit documents for an employee."""
    try:
        supabase = get_supabase_client()
        response = benefit_agent.process_request(
            req, 
            supabase_client=supabase,
            get_signed_url=gcs_manager.get_signed_url,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            check_file_exists=gcs_manager.check_file_exists
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching benefits: {str(e)}")

@app.post("/policy/fetch", response_model=PolicyFetchResponse)
async def fetch_policies(req: PolicyFetchRequest):
    """Fetch and summarize policy documents for an employee."""
    try:
        supabase = get_supabase_client()
        response = policy_agent.process_request(
            req, 
            supabase_client=supabase,
            get_signed_url=gcs_manager.get_signed_url,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            check_file_exists=gcs_manager.check_file_exists
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching policies: {str(e)}")

@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    """Chat endpoint for policy and benefits questions"""
    try:
        response = chat_agent.process_chat_request(
            question=req.question,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            get_signed_url=gcs_manager.get_signed_url,
            check_file_exists=gcs_manager.check_file_exists
        )
        
        return {
            "response": response,
            "status": "success",
            "user_id": req.user_id,
            "session_id": req.session_id
        }
    
    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error while processing your request. Please try again later.",
            "status": "error",
            "error": str(e)
        }

@app.get("/chat/health")
async def chat_health():
    """Health check for chat service"""
    return {
        "status": "healthy",
        "service": "chat_agent",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/chat/documents")
async def list_available_documents():
    """List all available documents in the GCS bucket"""
    try:
        policy_files = gcs_manager.list_files_in_folder("policies/")
        benefit_files = gcs_manager.list_files_in_folder("benefits/")
        
        documents = {
            "policies": [f.split('/')[-1] for f in policy_files if f.endswith('.pdf')],
            "benefits": [f.split('/')[-1] for f in benefit_files if f.endswith('.pdf')],
            "total_count": len([f for f in policy_files if f.endswith('.pdf')]) + len([f for f in benefit_files if f.endswith('.pdf')])
        }
        
        return {
            "documents": documents,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "AI Onboarding System - Multiple Agents",
        "status": "running",
        "available_endpoints": [
            "/health - Health check",
            "/welcome - Generate welcome message",
            "/onboard - Complete onboarding process",
            "/benefit/fetch - Fetch benefits",
            "/policy/fetch - Fetch policies",
            "/docs - API documentation"
        ],
        "agents": {
            "welcome_agent": "✅ Active",
            "benefit_agent": "✅ Active",
            "policy_agent": "✅ Active"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Onboarding System...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🎯 All agents active!")
    uvicorn.run(app, host="0.0.0.0", port=8000)