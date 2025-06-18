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
from contextlib import asynccontextmanager

# Import our agents
from agents.policy_agent import PolicyAgent, PolicyFetchRequest, PolicyFetchResponse
from agents.benefit_agent import BenefitAgent, BenefitFetchRequest, BenefitFetchResponse
from agents.policy_agent import PolicyAgent, PolicyFetchRequest, PolicyFetchResponse
from agents.expense_agent import ExpenseAgent, ExpenseFetchRequest, ExpenseFetchResponse
from agents.welcome_agent import WelcomeAgent, WelcomeRequest, WelcomeResponse
from agents.startup_vector_agent import StartupVectorChatAgent
from agents.onboarding_agent import OnboardingAgent, OnboardingFetchResponse, OnboardingFetchRequest

# Import GCS utilities
from utils.gcs_utils import GCSManager

# Load environment variables
load_dotenv()

class ChatRequest(BaseModel):
    email: str
    question: str

# New models for document acknowledgment
class DocumentAcknowledgmentRequest(BaseModel):
    email: str
    document_name: str
    # ✅ Removed acknowledged field since it doesn't exist in database

class DocumentAcknowledgmentResponse(BaseModel):
    success: bool
    message: str
    acknowledged: bool = False
    acknowledgment_id: Optional[str] = None
    timestamp: Optional[str] = None

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

onboarding_specialist = Agent(
    role='Onboarding Specialist',
    goal='Guide new employees through their onboarding process effectively',
    backstory="""You are an expert onboarding specialist who helps new employees navigate 
    their first days and weeks at the company. You excel at breaking down complex onboarding 
    processes into clear, manageable steps. You understand the importance of a smooth transition 
    and focus on providing practical guidance and support. You're particularly skilled at 
    customizing the onboarding experience based on role and department.""",
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
onboarding_agent = OnboardingAgent(agent=onboarding_specialist)
expense_agent = ExpenseAgent(agent=expense_specialist)
chat_agent = StartupVectorChatAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
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
    
    yield
    
    # Shutdown (cleanup if needed)
    print("🛑 Server shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="AI Onboarding System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    """Chat endpoint for policy and benefits questions"""
    try:
        supabase = get_supabase_client()
        response = chat_agent.process_chat_request(
            question=req.question,
            email=req.email,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            get_signed_url=gcs_manager.get_signed_url,
            check_file_exists=gcs_manager.check_file_exists,
            supabase_client=supabase
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
            "/welcome/fetch - Generate welcome message",
            "/benefit/fetch - Fetch benefits",
            "/policy/fetch - Fetch policies",
            "/expense/fetch - Fetch expense policies",
            "/onboard/agent - Fetch onboarding task details",
            "/chat - Chat with HR assistant",
            "/chat/documents - List available documents",
            "/document/acknowledge - Record document acknowledgment",
            "/document/acknowledgments/{email} - Get user acknowledgments",
            "/document/acknowledgments - Get all acknowledgments",
            "/docs - API documentation"
        ],
        "agents": {
            "welcome_agent": "✅ Active",
            "benefit_agent": "✅ Active",
            "policy_agent": "✅ Active",
            "expense_agent": "✅ Active",
            "onboarding_agent": "✅ Active",  # Updated to include onboarding agent
            "chat_agent": "✅ Active"
        }
    }

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

# Add the new onboarding tasks endpoint
@app.post("/onboard/fetch", response_model= OnboardingFetchResponse)
async def get_onboarding_tasks(req: OnboardingFetchRequest):
    """Fetch personalized onboarding task information for a new employee."""
    try:
        supabase = get_supabase_client()
        response = onboarding_agent.process_request(
            req,
            supabase_client=supabase,
            get_signed_url=gcs_manager.get_signed_url,
            list_files_in_folder=gcs_manager.list_files_in_folder,
            check_file_exists=gcs_manager.check_file_exists
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting onboarding tasks: {str(e)}")

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

@app.get("/chat/documents")
async def list_available_documents():
    """List all available documents in the GCS bucket"""
    try:
        # Updated to match your nested folder structure
        policy_files = gcs_manager.list_files_in_folder("onboarding agent/policy/")
        benefit_files = gcs_manager.list_files_in_folder("onboarding agent/benefits/")
        expense_files = gcs_manager.list_files_in_folder("onboarding agent/expense/")
        onboarding_files = gcs_manager.list_files_in_folder("onboarding agent/onboarding/")

        documents = {
            "policies": [f.split('/')[-1] for f in policy_files if f.endswith('.pdf')],
            "benefits": [f.split('/')[-1] for f in benefit_files if f.endswith('.pdf')],
            "expenses": [f.split('/')[-1] for f in expense_files if f.endswith('.pdf')],
            "onboarding": [f.split('/')[-1] for f in onboarding_files if f.endswith('.pdf')],
            "total_count": len([f for f in policy_files if f.endswith('.pdf')]) + 
                          len([f for f in benefit_files if f.endswith('.pdf')]) +
                          len([f for f in expense_files if f.endswith('.pdf')]) +
                          len([f for f in onboarding_files if f.endswith('.pdf')])
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
#To save the acknowledge document in db
@app.post("/document/acknowledge", response_model=DocumentAcknowledgmentResponse)
async def acknowledge_document(req: DocumentAcknowledgmentRequest):
    """
    Record that a user has acknowledged reading a document.
    
    This endpoint stores the acknowledgment in Supabase with:
    - User email
    - Document name
    - Timestamp
    """
    try:
        supabase = get_supabase_client()
          # Prepare the acknowledgment data
        acknowledgment_data = {
            "email": req.email,
            "document_name": req.document_name,
            "acknowledged_at": datetime.datetime.now().isoformat(),
            "created_at": datetime.datetime.now().isoformat()
            # ✅ Removed the acknowledged field since it doesn't exist in DB
        }
        
        # Insert into the document_acknowledgments table
        response = supabase.table("document_acknowledgments").insert(acknowledgment_data).execute()
        
        if response.data:
            acknowledgment_id = response.data[0].get("id") if response.data[0] else None            
            return DocumentAcknowledgmentResponse(
                success=True,
                acknowledged=True,  # ✅ Set to True in response since user is acknowledging
                message=f"Document '{req.document_name}' acknowledgment recorded successfully",
                acknowledgment_id=str(acknowledgment_id) if acknowledgment_id else None,
                timestamp=acknowledgment_data["acknowledged_at"]
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to record document acknowledgment"
            )
            
    except Exception as e:
        print(f"Error recording document acknowledgment: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error recording document acknowledgment: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Onboarding System...")
    print("📍 Server will be available at: http://localhost:4001")
    print("📚 API documentation at: http://localhost:4001/docs")
    print("🎯 All agents active!")
    uvicorn.run(app, host="0.0.0.0", port=4001)
