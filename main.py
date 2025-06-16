from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging

# Import our agents - using simplified Welcome Agent
from agents.welcome_agent_simple import SimpleWelcomeAgent, WelcomeRequest, WelcomeResponse

# Import GCS utilities
from utils.gcs_utils import GCSManager

# Try to import optional dependencies
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

# Try to import benefit agent (optional for now)
try:
    from agents.benefit_agent import BenefitAgent, BenefitFetchRequest, BenefitFetchResponse
    BENEFIT_AGENT_AVAILABLE = True
except ImportError:
    BENEFIT_AGENT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Onboarding System - Integrated Welcome Agent")

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
    benefits_summary: str
    next_steps: list
    documents: list
    team_contacts: dict

# Initialize helpers
gcs_manager = GCSManager()

def get_supabase_client():
    """Create and return a Supabase client."""
    if not SUPABASE_AVAILABLE:
        return None
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        return create_client(supabase_url, supabase_key)
    return None

# Initialize Welcome Agent (always available)
welcome_agent = SimpleWelcomeAgent()
logger.info("✅ Welcome Agent initialized")

# Initialize Benefit Agent (if available)
if BENEFIT_AGENT_AVAILABLE:
    try:
        benefit_agent = BenefitAgent()
        logger.info("✅ Benefit Agent initialized")
    except Exception as e:
        logger.warning(f"❌ Benefit Agent failed to initialize: {e}")
        BENEFIT_AGENT_AVAILABLE = False
else:
    logger.info("ℹ️ Benefit Agent not available - continuing with Welcome Agent only")

# Log available services
logger.info(f"🔧 Supabase available: {SUPABASE_AVAILABLE}")
logger.info(f"🔧 Benefit Agent available: {BENEFIT_AGENT_AVAILABLE}")

@app.post("/onboard", response_model=OnboardingResponse)
async def onboard_employee(req: OnboardingRequest):
    """
    Complete onboarding process using integrated Welcome Agent
    
    This endpoint orchestrates agents to provide a comprehensive onboarding experience:
    1. Welcome Agent: Creates personalized welcome message and guidance
    2. Benefits Agent: Fetches and summarizes benefit documents (if available)
    """
    try:
        logger.info(f"🎯 Processing onboarding for {req.name}")
        
        # Get detailed welcome info from Welcome Agent
        welcome_request = WelcomeRequest(
            name=req.name,
            role=req.role,
            team=req.team,
            manager=req.manager,
            start_date=req.start_date,
            email=req.email
        )
        welcome_response = welcome_agent.generate_welcome_message(welcome_request)
        
        # Get benefits information (if benefit agent is available)
        benefits_summary = "Benefits information will be provided by HR during your first week."
        if BENEFIT_AGENT_AVAILABLE:
            try:
                supabase = get_supabase_client()
                if supabase:
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
                    else:
                        logger.warning("Benefits processing failed, using fallback")
                else:
                    logger.warning("Supabase client not available")
            except Exception as e:
                logger.error(f"Benefits processing error: {e}")
        
        return OnboardingResponse(
            success=True,
            welcome_message=welcome_response.welcome_message if welcome_response.success else f"Welcome {req.name}! We're excited to have you join our {req.team} team as our new {req.role}.",
            benefits_summary=benefits_summary,
            next_steps=welcome_response.next_steps if welcome_response.success else [
                f"Contact your manager {req.manager} for guidance",
                "Complete employee documentation",
                "Review company policies"
            ],
            documents=welcome_response.available_documents if welcome_response.success else [
                "Employee Handbook", "Company Policies"
            ],
            team_contacts=welcome_response.team_contacts if welcome_response.success else {
                "manager": req.manager,
                "team": req.team
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Onboarding error for {req.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing onboarding: {str(e)}")

@app.post("/welcome", response_model=WelcomeResponse)
async def generate_welcome(req: WelcomeRequest):
    """Generate personalized welcome message."""
    try:
        response = welcome_agent.generate_welcome_message(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating welcome message: {str(e)}")

@app.post("/chat")
async def chat_with_welcome_agent(question: str, user_name: str):
    """Chat with the welcome agent."""
    try:
        answer = welcome_agent.answer_question(question, user_name)
        return {"answer": answer, "user": user_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Conditionally add benefit endpoint only if benefit agent is available
if BENEFIT_AGENT_AVAILABLE:
    @app.post("/benefit/fetch", response_model=BenefitFetchResponse)
    async def fetch_benefits(req: BenefitFetchRequest):
        """Fetch and summarize benefit documents for an employee."""
        try:
            supabase = get_supabase_client()
            if not supabase:
                raise HTTPException(status_code=503, detail="Database service unavailable")
                
            response = benefit_agent.process_request(
                req, 
                supabase_client=supabase,
                get_signed_url=gcs_manager.get_signed_url,
                list_files_in_folder=gcs_manager.list_files_in_folder,
                check_file_exists=gcs_manager.check_file_exists
            )
            return response
        except Exception as e:
            logger.error(f"❌ Benefits fetch error: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching benefits: {str(e)}")
else:
    @app.post("/benefit/fetch")
    async def fetch_benefits_unavailable():
        """Benefit service unavailable endpoint."""
        raise HTTPException(
            status_code=503, 
            detail="Benefit Agent service is currently unavailable. Please contact HR for benefits information."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    welcome_health = welcome_agent.health_check()
    
    health_status = {
        "status": "healthy",
        "message": "Welcome Agent integration successful",
        "agents": {
            "welcome_agent": welcome_health
        },
        "version": "1.0.0-integrated"
    }
    
    # Add benefit agent status if available
    if BENEFIT_AGENT_AVAILABLE:
        health_status["agents"]["benefit_agent"] = {
            "status": "healthy", 
            "agent_type": "benefit",
            "available": True
        }
    else:
        health_status["agents"]["benefit_agent"] = {
            "status": "unavailable", 
            "agent_type": "benefit",
            "available": False,
            "reason": "Dependencies not available"
        }
    
    return health_status

# Development endpoints for testing
@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "AI Onboarding System - Welcome Agent Integrated",
        "status": "running",
        "available_endpoints": [
            "/health - Health check",
            "/welcome - Generate welcome message",
            "/chat - Chat with welcome agent", 
            "/onboard - Complete onboarding process",
            "/benefit/fetch - Fetch benefits (if available)",
            "/docs - API documentation"
        ],
        "welcome_agent": "✅ Active",
        "benefit_agent": "🔄 Optional (check /health for status)"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Onboarding System with Integrated Welcome Agent...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("🎯 Welcome Agent integration active!")
    uvicorn.run(app, host="0.0.0.0", port=8000)