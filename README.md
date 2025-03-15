# Company Information Retrieval API

A FastAPI application that provides company information using a LangGraph-based agent workflow.

## WORKFLOW :

![graph_visualization](https://github.com/user-attachments/assets/1e7fb345-564d-44f3-a042-499a5e887075)

## Example Langsmith :

<img width="423" alt="image" src="https://github.com/user-attachments/assets/7e0c03e6-1c4b-4e0e-92f6-1e26315a5d20" />

## Features

- Query information about companies using natural language
- Async processing with background tasks
- Containerized deployment with Docker
- Uses LangGraph for workflow orchestration
- Integrates with Tavily and Wikipedia for information retrieval

## Setup and Installation

### Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key
- `LANGCHAIN_API_KEY`: Your LangSmith API key

### Running with Docker

1. Build the Docker image:

```bash
docker build -t company-info-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 --env-file .env company-info-api
```

The API will be available at http://localhost:8000

### Running Locally

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:

```bash
uvicorn app:app --reload
```

## API Endpoints

### Submit a query
`POST /query`

Request body:
```json
{
  "query": "Who is the CEO of OpenAI?",
  "thread_id": "optional-thread-id-for-conversation-context"
}
```

Response:
```json
{
  "request_id": "unique-request-id",
  "status": "processing",
  "thread_id": "thread-id"
}
```

### Get query results
`GET /result/{request_id}`

Response:
```json
{
  "request_id": "unique-request-id",
  "status": "completed",
  "result": {
    "response": "The CEO of OpenAI is Sam Altman.",
    "references": ["reference1", "reference2"]
  }
}
```

### Health check
`GET /health`

Response:
```json
{
  "status": "healthy"
}
```

## Development

### Project Structure

- `app.py`: FastAPI application
- `graph.py`: LangGraph workflow implementation
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies

## Deployment

This application can be deployed to any container orchestration platform like Kubernetes, ECS, or simply run as a container on a VM. 
