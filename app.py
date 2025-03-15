import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from uuid import uuid4

# Import the graph and necessary components from your existing code
from graph_test_v2 import (
    graph, OverallState, 
    # Add any other necessary imports
)

# Create a dictionary to store results
results_store = {}

class QueryRequest(BaseModel):
    query: str
    thread_id: str = None

class QueryResponse(BaseModel):
    request_id: str
    status: str
    thread_id: str = None

class ResultResponse(BaseModel):
    request_id: str
    status: str
    result: Dict[str, Any] = None
    error: str = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: load models, initialize components, etc.
    print("Starting up FastAPI server...")
    yield
    # Cleanup: release resources
    print("Shutting down FastAPI server...")

app = FastAPI(lifespan=lifespan, title="Company Information Agent API")

async def process_query(request_id: str, user_query: str, thread_id: str = None):
    """Background task to process the query through the LangGraph workflow"""
    try:
        # Configure thread if provided or create a new one
        if thread_id:
            thread = {"configurable": {"thread_id": thread_id}}
        else:
            thread = {"configurable": {"thread_id": str(uuid4())}}
        
        # Run the graph and collect results
        final_result = None
        async for event in graph.astream(
            {"initial_query": user_query}, 
            thread, 
            stream_mode="values"
        ):
            # Keep track of the latest event
            final_result = event
        
        # Extract relevant information from the final result
        response = ""
        references = []
        
        if final_result and "response" in final_result:
            response = final_result["response"]
        
        # Check if final_result includes references from prepare_output
        if final_result and "references" in final_result:
            references = final_result["references"]
        
        # Store the final result and thread ID
        results_store[request_id] = {
            "status": "completed",
            "result": {
                "response": response,
                "references": references
            },
            "thread_id": thread["configurable"]["thread_id"]
        }
    except Exception as e:
        # Handle errors
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        
        results_store[request_id] = {
            "status": "error",
            "error": str(e),
            "error_details": error_details,
            "thread_id": thread_id
        }

@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a query to the agent for processing"""
    request_id = str(uuid4())
    
    # Initialize the request in the results store
    results_store[request_id] = {
        "status": "processing",
        "thread_id": request.thread_id
    }
    
    # Schedule the processing in a background task
    background_tasks.add_task(
        process_query, 
        request_id, 
        request.query, 
        request.thread_id
    )
    
    return QueryResponse(
        request_id=request_id,
        status="processing",
        thread_id=request.thread_id
    )

@app.get("/result/{request_id}", response_model=ResultResponse)
async def get_result(request_id: str):
    """Get the result of a previously submitted query"""
    if request_id not in results_store:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    result_data = results_store[request_id]
    
    return ResultResponse(
        request_id=request_id,
        status=result_data["status"],
        result=result_data.get("result"),
        error=result_data.get("error")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 