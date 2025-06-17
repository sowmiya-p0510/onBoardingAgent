# Onboarding Agent System

## Overview

The Onboarding Agent System is a sophisticated platform designed to streamline the employee onboarding process by providing role-specific information about company policies, benefits, and expense procedures. The system uses intelligent agents to fetch, process, and deliver relevant documentation to new employees based on their role within the organization.

## Key Features

- **Role-based Document Retrieval**: Automatically fetches documents relevant to an employee's specific role
- **Document Management**: Organizes policies, benefits, and expense documentation in a structured format
- **Secure Document Access**: Provides secure, temporary access to sensitive company documents
- **Optimized Performance**: Uses pre-generated summaries to deliver information efficiently

## Tech Stack

### Backend
- **Python**: Core programming language for the application logic
- **FastAPI**: High-performance web framework for building APIs
- **Pydantic**: Data validation and settings management
- **Supabase**: Database and authentication services
- **Google Cloud Storage (GCS)**: Document storage and management

### Infrastructure
- **Docker**: Containerization for consistent deployment
- **AWS**: Cloud hosting and infrastructure
- **AWS Lambda**: Serverless execution environment
- **AWS ECS/Fargate**: Containerized application deployment
- **AWS API Gateway**: API management and routing

### Document Processing
- **Document Models**: Structured representation of company documents
- **URL Signing**: Secure, time-limited access to documents

### API Components
- **Request/Response Models**: Standardized data exchange format
- **Agent Classes**: Specialized handlers for different document types

### Development Tools
- **Git**: Version control
- **GitHub Actions**: CI/CD pipeline
- **Poetry**: Python dependency management
- **Pytest**: Testing framework

## Agent Architecture

The system comprises three specialized agents:

1. **Onboarding Agent**: Retrieves and processes company onboarding documents relevant to an employee's role
2. **Benefit Agent**: Provides information about benefits available to employees based on their position
3. **Expense Agent**: Delivers expense policies, procedures, and guidelines specific to each role

Each agent follows a similar workflow:
1. Receive a request with employee role information
2. Query the database for relevant documents
3. Generate secure access URLs for the documents
4. Format and return document information with pre-generated summaries

## Data Structure

Documents are stored with the following attributes:

- **Title**: Name of the document
- **URL**: Secure access link to the document
- **Description**: Brief description of the document's content
- **Summary**: Pre-generated concise summary of key points

## Implementation Details

- Documents are stored in Google Cloud Storage (GCS) and referenced in Supabase
- Application hosting and deployment is managed on AWS infrastructure
- Each agent queries a specific table in Supabase filtered by role and agent type
- Document summaries are pre-generated and stored in the database to optimize performance
- Secure URLs are generated on-demand when documents are requested using GCS signed URLs

## Usage

The system is designed to be integrated into a larger onboarding platform, with each agent handling specific types of information requests. When a new employee joins, the appropriate agents are triggered to provide relevant documentation based on the employee's role.

### Example Code

```python
from agents.expense_agent import ExpenseAgent, ExpenseFetchRequest
from agents.onboarding_agent import OnboardingAgent, OnboardingFetchRequest
from your_supabase_client import supabase_client
from your_storage_utils import get_signed_url, list_files_in_folder, check_file_exists

# Initialize the onboarding agent
onboarding_agent = OnboardingAgent(agent=None)

# Create a request for a specific role
onboarding_request = OnboardingFetchRequest(role="manager", email="new.manager@company.com")

# Process the request
onboarding_response = onboarding_agent.process_request(
    req=onboarding_request,
    supabase_client=supabase_client,
    get_signed_url=get_signed_url,
    list_files_in_folder=list_files_in_folder,
    check_file_exists=check_file_exists
)

# Handle the response
if onboarding_response.success:
    print(f"Found {len(onboarding_response.documents)} onboarding documents")
    for doc in onboarding_response.documents:
        print(f"Document: {doc.title}")
        print(f"Access URL: {doc.url}")
        print(f"Acknowledged: {doc.acknowledged}")
else:
    print(f"Error: {onboarding_response.message}")
```

## Database Schema

### Supabase Tables

#### agents Table
- `id`: Unique identifier
- `role`: Employee role (e.g., manager, developer)
- `agent_type`: Type of agent (onboarding, benefit, expense)
- `documents`: JSON array containing document metadata:
  - `doc_title`: Document title
  - `gcs_url`: Storage folder path
  - `description`: Document description
  - `summary`: Pre-generated document summary
  - `pdf_name`: Filename of the document

## Benefits

- **Efficiency**: Reduces manual effort in onboarding by automating document delivery
- **Consistency**: Ensures all employees receive standardized information
- **Personalization**: Tailors information to each employee's specific role
- **Accessibility**: Provides easy access to important company information
- **Scalability**: Easily accommodates new roles and document types

## Getting Started

### Prerequisites

- Python 3.8+
- AWS account for application hosting
- Google Cloud Platform account for document storage
- Supabase account

### Installation

```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd onBoardingAgent

# Install dependencies using Poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### Configuration

1. **Google Cloud Storage Setup**:
   - Create GCS buckets for document storage
   - Configure appropriate permissions and access controls
   - Set up service accounts for application access

2. **AWS Deployment**:
   - Configure AWS Lambda functions or ECS services for application hosting
   - Set up API Gateway for API endpoints
   - Configure IAM roles with appropriate permissions
   - Deploy containerized application to AWS services

3. **Supabase Setup**:
   - Set up your Supabase project and create the necessary tables
   - Configure authentication and storage policies
   - Set environment variables for API keys and connection strings
