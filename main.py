from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Optional
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
import datetime 

# Import our agents
from agents.benefit_agent import BenefitAgent, BenefitFetchRequest, BenefitFetchResponse
from agents.policy_agent import PolicyAgent, PolicyFetchRequest, PolicyFetchResponse
from agents.expense_agent import ExpenseAgent, ExpenseFetchRequest, ExpenseFetchResponse
from agents.welcome_agent import WelcomeAgent, WelcomeRequest, WelcomeResponse, UserProfile
from agents.startup_vector_agent import StartupVectorChatAgent

# Import GCS utilities
from utils.gcs_utils import GCSManager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Onboarding System")

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
    expense_summary: str = None
    next_steps: list
    documents: list
    team_contacts: dict

class ChatRequest(BaseModel):
    email: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

expense_specialist = Agent(
    role='Expense Management Specialist',
    goal='Help employees understand and navigate expense policies and procedures',
    backstory="""You are an experienced expense management specialist with deep knowledge of 
    corporate expense policies and reimbursement processes. You excel at explaining expense 
    procedures in simple, actionable terms. You understand that employees need clear guidance 
    on submitting expenses correctly the first time, so you focus on practical explanations 
    and common pitfalls to avoid. You're particularly skilled at highlighting expense limits, 
    required documentation, and approval workflows. You always emphasize best practices for 
    efficient expense management and timely reimbursement.""",
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
welcome_agent = WelcomeAgent(agent=hr_agent)
benefit_agent = BenefitAgent(agent=benefit_specialist)
policy_agent = PolicyAgent(agent=policy_specialist)
expense_agent = ExpenseAgent(agent=expense_specialist)
chat_agent = StartupVectorChatAgent()

@app.on_event("startup")
async def startup_event():
    """Preload vector store when server starts"""
    print("🚀 Server starting - initializing chat agent...")
    
    # Set GCS functions for the chat agent
    chat_agent.set_gcs_functions(
        gcs_manager.list_files_in_folder,
        gcs_manager.get_signed_url,
        gcs_manager.check_file_exists
    )
    
    # Start background preloading
    preload_thread = chat_agent.preload_vector_store_background()
    print("📋 Vector store preloading started in background...")
    print("🎯 Server ready! Vector store will be available shortly.")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "All agents operational",
        "agents": {
            "welcome_agent": {"status": "healthy", "agent_type": "welcome"},
            "benefit_agent": {"status": "healthy", "agent_type": "benefit"},
            "policy_agent": {"status": "healthy", "agent_type": "policy"},
            "expense_agent": {"status": "healthy", "agent_type": "expense"},
            "chat_agent": {"status": "healthy", "agent_type": "chat"}
        },
        "version": "1.0.0"
    }

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
            "email": req.email
        }

    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error while processing your request. Please try again later.",
            "status": "error",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "AI Onboarding System - Multiple Agents",
        "status": "running",
        "available_endpoints": [
            "/health - Health check",
            "/welcome/fetch - Generate welcome message",
            "/onboard - Complete onboarding process",
            "/benefit/fetch - Fetch benefits",
            "/policy/fetch - Fetch policies",
            "/expense/fetch - Fetch expense policies",
            "/chat - Chat with HR assistant",
            "/chat/health - Chat service health check",
            "/docs - API documentation"
        ],
        "agents": {
            "welcome_agent": "✅ Active",
            "benefit_agent": "✅ Active",
            "policy_agent": "✅ Active",
            "expense_agent": "✅ Active",
            "chat_agent": "✅ Active"
        }
    }

@app.post("/onboard", response_model=OnboardingResponse)
async def onboard_employee(req: OnboardingRequest):
    """
    Complete onboarding process using multiple agents

    This endpoint orchestrates agents to provide a comprehensive onboarding experience:
    1. Welcome Agent: Creates personalized welcome message and guidance
    2. Benefits Agent: Fetches and summarizes benefit documents
    3. Policy Agent: Fetches and summarizes policy documents
    4. Expense Agent: Fetches and summarizes expense documents
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

        # Get expense information
        expense_summary = "Expense policies and procedures will be provided during your orientation."
        try:
            expense_request = ExpenseFetchRequest(role=req.role, email=req.email)
            expense_response = expense_agent.process_request(
                expense_request,
                supabase_client=supabase,
                get_signed_url=gcs_manager.get_signed_url,
                list_files_in_folder=gcs_manager.list_files_in_folder,
                check_file_exists=gcs_manager.check_file_exists
            )
            if expense_response.success:
                expense_summary = expense_response.overall_summary
        except Exception as e:
            print(f"Expense processing error: {e}")

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
            "Review company policies",
            "Set up your expense management account"
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
            expense_summary=expense_summary,
            next_steps=next_steps,
            documents=["Employee Handbook", "Company Policies", "Expense Guidelines"],
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

@app.post("/expense/fetch", response_model=ExpenseFetchResponse)
async def fetch_expenses(req: ExpenseFetchRequest):
    """Fetch and summarize expense documents for an employee."""
    try:
        supabase = get_supabase_client()
        response = expense_agent.process_request(
            req, 
            supabase_client=supabase,
            get_signed_url=gcs_manager.get_signed_url,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            check_file_exists=gcs_manager.check_file_exists
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching expense documents: {str(e)}")

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
        expense_files = gcs_manager.list_files_in_folder("expenses/")

        documents = {
            "policies": [f.split('/')[-1] for f in policy_files if f.endswith('.pdf')],
            "benefits": [f.split('/')[-1] for f in benefit_files if f.endswith('.pdf')],
            "expenses": [f.split('/')[-1] for f in expense_files if f.endswith('.pdf')],
            "total_count": len([f for f in policy_files if f.endswith('.pdf')]) + 
                          len([f for f in benefit_files if f.endswith('.pdf')]) +
                          len([f for f in expense_files if f.endswith('.pdf')])
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Onboarding System...")
    print("📍 Server will be available at: http://localhost:4001")
    print("📚 API documentation at: http://localhost:4001/docs")
    print("🎯 All agents active!")
    uvicorn.run(app, host="0.0.0.0", port=4001)
