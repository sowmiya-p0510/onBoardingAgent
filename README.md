# Onboarding Agent

A FastAPI application that uses CrewAI and LangChain to create an onboarding assistant for new employees.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env` and add required API keys and credentials
   ```
   # OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Supabase credentials
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_key_here
   
   # Google Cloud Storage credentials
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   ```
4. Set up Google Cloud Storage:
   - Create a GCS bucket to store benefit documents
   - Store your Google Cloud credentials JSON file securely
   
5. Set up Supabase:
   - Create a 'documents' table with columns:
     - title (text)
     - url (text, format: gs://bucket-name/path/to/file)
     - agent_type (text, set as 'benefit' for benefit documents)

## Running the Application

Run the server:

```
python main.py
```

Or directly with uvicorn:

```
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Usage

### Onboarding Endpoint

Send a POST request to `/onboard` with the following JSON structure:

```json
{
  "name": "John Doe",
  "role": "Software Engineer",
  "start_date": "June 15, 2025"
}
```

### Benefits Endpoint

Send a POST request to `/benefits` with one of the following JSON structures:

For a benefits summary:
```json
{
  "role": "Software Engineer"
}
```

For answering a specific benefits question:
```json
{
  "role": "Software Engineer",
  "question": "What health insurance options do I have?"
}
```

## API Documentation

Once running, access the auto-generated API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
