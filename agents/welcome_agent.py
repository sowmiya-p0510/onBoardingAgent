from typing import Optional
from pydantic import BaseModel
from crewai import Task, Crew
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
    def __init__(self, agent):
        """
        Initialize the WelcomeAgent with a pre-defined agent.

        Args:
            agent: A CrewAI Agent instance to use for processing
        """
        self.agent = agent

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

            # Generate welcome message
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
        """Generate a personalized welcome message for the user."""
        # Format joining date for better readability
        try:
            joining_date = datetime.datetime.strptime(user_profile.joining_date, "%Y-%m-%d")
            formatted_joining_date = joining_date.strftime("%B %d, %Y")
        except:
            formatted_joining_date = user_profile.joining_date

        welcome_task = Task(
            description=f"""
            Create a warm, personalized welcome message for a new employee with the following details:

            Full Name: {user_profile.full_name}
            Role: {user_profile.role}
            Department: {user_profile.department}
            Manager: {user_profile.manager_name}
            Joining Date: {formatted_joining_date}
            Location: {user_profile.location}
            Employment Type: {user_profile.employment_type}

            The welcome message should:
            1. Greet the employee by name
            2. Welcome them to the company and mention their specific role and department
            3. Express enthusiasm about their joining
            4. Mention their manager by name and encourage reaching out for support
            5. Reference their joining date and location
            6. End with a positive note about looking forward to their contributions

            IMPORTANT FORMATTING INSTRUCTIONS:
            - DO NOT include any signature, sign-off, or closing line like "Best regards," "Sincerely," etc.
            - DO NOT include any placeholders like "[Your Name]" or "[Your Position]"
            - End the message directly after the final paragraph with no additional text

            The tone should be professional, warm, enthusiastic, and human-like.
            DO NOT include any references to AI, automation, or this being a generated message.
            Write as if you are a human HR representative or team leader genuinely welcoming a new colleague.

            The message should be 4-6 paragraphs long and feel sincere and personalized.
            """,
            expected_output="A warm, professional, and personalized welcome message with no signature or placeholders",
            agent=self.agent
        )

        # Create the Crew for welcome message
        crew = Crew(
            agents=[self.agent],
            tasks=[welcome_task],
            verbose=False
        )

        # Run the crew and convert output to string
        crew_result = crew.kickoff()
        if hasattr(crew_result, 'raw'):
            welcome_message = str(crew_result.raw)
        else:
            welcome_message = str(crew_result)

        return welcome_message