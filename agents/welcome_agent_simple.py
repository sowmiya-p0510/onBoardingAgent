#!/usr/bin/env python3
"""
Simplified Welcome Agent using direct OpenAI API calls
This version avoids complex CrewAI dependencies while providing the same functionality
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import openai
from openai import OpenAI
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models for Welcome Agent
class WelcomeRequest(BaseModel):
    """Request model for welcome message generation"""
    name: str = Field(..., description="Employee's full name")
    role: str = Field(..., description="Job role/title")
    team: str = Field(..., description="Team/department name")
    manager: str = Field(..., description="Manager's name")
    start_date: str = Field(..., description="Start date")
    email: Optional[str] = Field(None, description="Email address")

class WelcomeResponse(BaseModel):
    """Response model for welcome message"""
    success: bool
    welcome_message: str
    next_steps: List[str]
    team_contacts: Dict[str, str]
    available_documents: List[str]

class SimpleWelcomeAgent:
    """
    Simplified Welcome Agent using direct OpenAI API calls
    Handles personalized welcome messages and onboarding guidance
    """
    
    def __init__(self):
        """Initialize Welcome Agent with OpenAI client"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ Simple Welcome Agent initialized")
    
    def generate_welcome_message(self, req: WelcomeRequest) -> WelcomeResponse:
        """
        Generate comprehensive welcome response for new employee
        
        Args:
            req: Welcome request with employee details
            
        Returns:
            Complete welcome response with message, next steps, and resources
        """
        try:
            logger.info(f"🎉 Generating welcome response for {req.name}")
            
            # Create personalized welcome message using OpenAI
            prompt = f"""
            Create a warm, personalized welcome message for a new employee:
            
            Name: {req.name}
            Role: {req.role}
            Team: {req.team}
            Manager: {req.manager}
            Start Date: {req.start_date}
            
            The message should:
            1. Personally address {req.name} by name
            2. Express genuine excitement about them joining as {req.role}
            3. Welcome them to the {req.team} team
            4. Mention their manager {req.manager} positively
            5. Reference their start date {req.start_date}
            6. Convey company culture and values of collaboration and growth
            7. Make them feel valued and excited about their journey ahead
            8. Be warm, professional, and encouraging
            9. Be 3-4 paragraphs long
            
            Make it feel personal and authentic, not templated.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly HR specialist who creates warm, personalized welcome messages for new employees. You understand company culture and make people feel valued and excited about joining the team."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            welcome_message = response.choices[0].message.content
            
            # Generate next steps
            next_steps = self._generate_next_steps(req)
            
            # Get team contacts
            team_contacts = self._get_team_contacts(req)
            
            # Get available documents
            available_documents = self._get_available_documents()
            
            response_obj = WelcomeResponse(
                success=True,
                welcome_message=welcome_message,
                next_steps=next_steps,
                team_contacts=team_contacts,
                available_documents=available_documents
            )
            
            logger.info(f"✅ Welcome response generated for {req.name}")
            return response_obj
            
        except Exception as e:
            logger.error(f"❌ Error generating welcome response: {e}")
            return WelcomeResponse(
                success=False,
                welcome_message=f"Welcome to the team, {req.name}! We're excited to have you join us as our new {req.role}. Your manager {req.manager} and the entire {req.team} team are looking forward to working with you starting {req.start_date}.",
                next_steps=["Contact your manager for guidance", "Check your email for onboarding instructions", "Review the employee handbook"],
                team_contacts={"manager": req.manager, "team": req.team},
                available_documents=["Employee Handbook", "Company Policies"]
            )
    
    def answer_question(self, question: str, user_name: str, context: str = "") -> str:
        """
        Answer onboarding questions using AI
        
        Args:
            question: User's question
            user_name: Name of person asking
            context: Additional context for better answers
            
        Returns:
            AI-generated answer
        """
        try:
            logger.info(f"💬 Answering question for {user_name}")
            
            prompt = f"""
            Answer this onboarding question from {user_name}:
            
            Question: {question}
            Additional Context: {context}
            
            Provide a helpful, accurate, and friendly response that:
            1. Directly addresses their question
            2. Is specific and actionable
            3. Maintains a welcoming tone
            4. Suggests next steps if appropriate
            5. Encourages them to ask follow-up questions
            
            If you don't have specific information, direct them to appropriate resources or people.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful onboarding assistant who answers employee questions in a friendly, informative way. You provide practical guidance and always maintain a supportive tone."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.6
            )
            
            answer = response.choices[0].message.content
            logger.info(f"✅ Question answered for {user_name}")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return f"I'm having trouble accessing information right now, {user_name}. Please reach out to your manager or HR for assistance with your question: '{question}'"
    
    def _generate_next_steps(self, req: WelcomeRequest) -> List[str]:
        """Generate personalized next steps for the employee"""
        base_steps = [
            f"Schedule an introductory meeting with your manager, {req.manager}",
            f"Meet your {req.team} team members and key stakeholders",
            "Complete your employee profile and emergency contact information",
            "Review the employee handbook and company policies",
            "Set up your workspace and access to necessary tools",
            "Complete any required training modules"
        ]
        
        # Add role-specific steps
        role_lower = req.role.lower()
        if 'engineer' in role_lower or 'developer' in role_lower:
            base_steps.extend([
                "Set up your development environment and access to code repositories",
                "Review the technical documentation and development guidelines"
            ])
        elif 'sales' in role_lower or 'marketing' in role_lower:
            base_steps.extend([
                "Review current campaigns and sales materials",
                "Schedule time with the sales/marketing team lead"
            ])
        elif 'manager' in role_lower or 'lead' in role_lower:
            base_steps.extend([
                "Review team structure and current projects",
                "Schedule one-on-ones with your direct reports"
            ])
        elif 'designer' in role_lower:
            base_steps.extend([
                "Review brand guidelines and design systems",
                "Access design tools and software licenses"
            ])
        
        return base_steps[:8]  # Return first 8 steps
    
    def _get_team_contacts(self, req: WelcomeRequest) -> Dict[str, str]:
        """Get relevant team contacts"""
        return {
            "manager": req.manager,
            "team": req.team,
            "hr_contact": "HR Department",
            "it_support": "IT Help Desk",
            "buddy_program": "Employee Buddy Program"
        }
    
    def _get_available_documents(self) -> List[str]:
        """Get list of available onboarding documents"""
        return [
            "Employee Handbook",
            "Company Policies",
            "Benefits Guide",
            "Org Chart",
            "Culture Deck",
            "First Week Checklist",
            "IT Setup Guide",
            "Emergency Contacts"
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        try:
            # Test OpenAI connection with a minimal request
            test_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "agent_type": "simple_welcome",
                "openai_connection": "active",
                "model": "gpt-3.5-turbo"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_type": "simple_welcome",
                "error": str(e)
            }

# Export classes
__all__ = ["SimpleWelcomeAgent", "WelcomeRequest", "WelcomeResponse"]
