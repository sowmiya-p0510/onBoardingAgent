from typing import Optional
from pydantic import BaseModel
import datetime

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
        pass

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

    def process_request(self, req: WelcomeRequest, supabase_client) -> WelcomeResponse:
        """Process a welcome message request."""
        try:
            # Get user profile data
            user_profile = self.get_user_profile(req.email, supabase_client)

            if not user_profile:
                return WelcomeResponse(
                    success=False,
                    message=f"No user profile found for email: {req.email}"
                )

            # Generate static welcome message
            welcome_message = self._generate_welcome_message(user_profile)

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

    def _generate_welcome_message(self, user_profile: UserProfile) -> str:
        """Generate a personalized welcome message using a template."""
        # Format joining date for better readability
        try:
            joining_date = datetime.datetime.strptime(user_profile.joining_date, "%Y-%m-%d")
            formatted_joining_date = joining_date.strftime("%B %d, %Y")
        except:
            formatted_joining_date = user_profile.joining_date

        # Extract first name for more personalized closing
        first_name = user_profile.full_name.split()[0] if user_profile.full_name else "there"

        # Template-based welcome message
        welcome_message = f"""Dear {user_profile.full_name},

We are thrilled to welcome you to our team as a {user_profile.employment_type} {user_profile.role} in the {user_profile.department} Department. It's an exciting time for us, and we're delighted that someone with your skills and enthusiasm is joining our team. We believe that with your abilities, we will reach new heights.

Starting a new job can be overwhelming, but remember, we are all here to support you. Your manager, {user_profile.manager_name}, is looking forward to working with you and is available to assist you in any way possible. Don't hesitate to reach out to him or any of your new colleagues if you have questions or need any help settling in.

We are eagerly awaiting your arrival on {formatted_joining_date}, at our {user_profile.location} location. We have planned a comprehensive onboarding process for you to ensure a smooth transition into your new role. This will help you understand our work culture, the projects you'll be working on, and how you can contribute to our shared goals.

We are confident that you will make significant contributions to our ongoing projects and future initiatives. We believe in your ability to bring fresh ideas and perspectives to our team, and we look forward to seeing your passion in action.

Once again, welcome aboard, {first_name}. We can't wait to start this exciting journey with you."""

        return welcome_message