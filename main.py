from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Import our BenefitAgent
from agents.benefit_agent import BenefitAgent, BenefitFetchRequest, BenefitFetchResponse
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

# Initialize helpers
gcs_manager = GCSManager()

def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)

# Initialize BenefitAgent with the defined agent
benefit_agent = BenefitAgent(agent=benefit_specialist)

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)