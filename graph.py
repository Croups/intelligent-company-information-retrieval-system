import os
import asyncio
from httpcore import Origin
import requests
import nest_asyncio
from typing import List, Optional, Dict, Any, TypedDict
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from tavily import AsyncTavilyClient
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import WikipediaRetriever

# Initialize environment
load_dotenv()
nest_asyncio.apply()

# No need to set them explicitly here

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class AmbiguityCheck(TypedDict):
    is_ambiguous: bool
    ambiguity_reason: Optional[str]
    clarification_question: Optional[str]
    
class InputAnalysis(TypedDict):
    company_name : str = Field(description="The name of the company the user is asking about")
    question_type : str = Field(description="The type of question the user is asking")
    
# Main state that tracks overall processing
class OverallState(TypedDict):
    # User query information
    initial_query : str
    input_analysis: Optional[InputAnalysis]
    user_clarification: Optional[str]
    ambiguity_check: Optional[AmbiguityCheck]

    retrieval_results: Optional[Dict[str, Any]]
    response : str
    
class Query(TypedDict):
    query : str
    
class Queries(TypedDict):
    queries : List[Query]
    
    
class Output(TypedDict):
    final_answer : str = Field(description="The final answer to the user's query")
    references : List[str] = Field(description="List of references to the sources")
    
class Evaluation(TypedDict):
    score : int = Field(description="The score of the if answer covers the question", le=3, ge=-3)
    reason : str = Field(description="The reason for the score")
    is_refinement_needed : bool = Field(description="Whether the search results need to be refined")
          
class WebSearchState(TypedDict, total=False):
    initial_query: str
    tavily_results: List[Dict[str, Any]]
    wikipedia_results: List[Dict[str, Any]]
    evaluation: Optional[Evaluation]
    final_answer: Optional[str]
    references: Optional[List[str]]

ambiguity_checker_instructions = """
You are an advanced AI assistant designed to analyze user queries about company information and determine if clarification is needed. Your task is to carefully consider the provided information and decide whether the query requires further clarification.

Here is the user's query:
<user_query>
{user_query}
</user_query>

Here is any human feedback provided (this may be empty):
<human_feedback>
{human_feedback}
</human_feedback>

Instructions:
1. Carefully analyze the user query and any provided human feedback.
2. Determine if the query needs clarification.
3. If clarification is needed, provide the reason for ambiguity, a clarification question, and an ambiguity score.
4. If no clarification is needed, indicate this in your response.
"""

input_analysis_instructions = """
You are an advanced AI assistant designed to analyze user queries about company information and determine the type of question and the company, user is asking about. 

Here is the user's query:
<user_query>
{user_query}
</user_query>

"""

query_generation_instructions = """
You are an advanced AI assistant designed to generate optimized queries for each data source.

Here is the initial query:
<initial_query>
{initial_query}
</initial_query>

"""

evaluation_instructions = """
You are an advanced AI assistant designed to evaluate the search results and determine if the answer covers the question.

Here is the answer:
<answer>
{answer}
</answer>

Here is the question:
<question>
{question}
</question>

Instructions:
1. Carefully analyze the answer and question.
2. Determine if the answer covers the question.
3. If the answer does not cover the question, provide the reason for the score.
4. If the answer covers the question, provide the score.
5. If the answer needs to be refined, provide the refinement strategy.
"""

output_instructions = """
You are an advanced AI assistant designed to prepare the final answer to the user's query from given sources.

Here is the wikipedia results:(for structured and well-documented information (company history,
fundamental details).)
<wikipedia_results>
{wikipedia_results}
</wikipedia_results>

Here is the tavily results:(for latest news, social media, and other non-structured information)
<tavily_results>
{tavily_results}
</tavily_results>

Instructions:
1. Prepare the final answer to the user's query from the given sources.
2. Include references to the sources in the final answer.
3. The final answer should be in a concise and informative manner.
"""

# Function comments detailing input/output and processing logic

def check_ambiguity(state: OverallState) -> Dict[str, str]:
    """
    Check if query needs clarification
    Input: state.input_analysis
    Output: Route to human_feedback or extract_user_intent
    """
    
    query = state["initial_query"]
    human_feedback = state.get("user_clarification", None)
    
    structured_llm = llm.with_structured_output(AmbiguityCheck)
        
    system_prompt = ambiguity_checker_instructions.format(user_query=query, human_feedback=human_feedback)
        
    print(system_prompt)

    result = structured_llm.invoke([SystemMessage(content=system_prompt),
                                HumanMessage(content="Based on the above instructions, please check if the user query needs clarification.")])    
    print(result)
    return {"ambiguity_check": result}
   

    
def analyze_query(state: OverallState) -> OverallState:
    """
    Extract company entity and intent from user query
    Input: state.initial_query
    Output: state.input_analysis with entity and intent information
    """
    
    query = state["initial_query"]
    
    structured_llm = llm.with_structured_output(InputAnalysis)
    
    system_prompt = input_analysis_instructions.format(user_query=query)
    
    print(system_prompt)

    result = structured_llm.invoke([SystemMessage(content=system_prompt),
                                HumanMessage(content="Based on the above instructions, please analyze the user query and determine the type of question and the company, user is asking about.")])    
    print(result)
    
    # Return both the input_analysis and ensure initial_query is passed to the subgraph
    return {
        "input_analysis": result,
        "initial_query": query  # Make sure initial_query is passed to the subgraph
    }


def route_after_ambiguity(state: OverallState) -> Dict[str, str]:
    
    is_ambiguous = state["ambiguity_check"]["is_ambiguous"]
    
    if is_ambiguous:
        return "human_feedback"
    else:
        return "analyze_query"

def human_feedback(state: OverallState) -> OverallState:
    """
    Get clarification from user
    Input: state with ambiguity details
    Output: state.user_clarification with user's response
    """
    interrupt_message = "Please provide additional information about the com"
    
    feedback = interrupt(interrupt_message)

    return {"user_clarification": feedback}

def generate_response(state: WebSearchState) -> WebSearchState:
    """
    Create final response for user
    Input: state containing search_results 
    Output: state.response with formatted answer
    """
    final_answer = state["final_answer"]
    
    return {"response": final_answer}


@traceable
async def tavily_search_async(search_queries):
    
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    include_answer=True,
                    topic="general"
                    
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs

@traceable
def wikipedia_search(initial_query: str) -> str:
    retriever = WikipediaRetriever()
    docs = retriever.invoke(initial_query) 

    return docs

def generate_queries(initial_query: str) -> Queries:
    """
    Generate optimized queries for each data source
    Input: initial_query
    Output: queries
    """
    
    structured_llm = llm.with_structured_output(Queries)
    
    system_prompt = query_generation_instructions.format(initial_query=initial_query)
    
    result = structured_llm.invoke([SystemMessage(content=system_prompt),
                                HumanMessage(content="Generate me queries for web search")])    
    print(result)
    return result

async def web_search_agent(state: WebSearchState) -> WebSearchState:
    """
    Execute queries against appropriate sources
    Input: state.initial_query
    Output: state.search_results with findings from each source
    """
    initial_query = state["initial_query"]
    
    queries = generate_queries(initial_query)
    
    queries = [query_dict['query'] for query_dict in queries["queries"]]  
    
    wikipedia_results = wikipedia_search(initial_query)  
    
    search_results = await tavily_search_async(queries)
    
    
    return {"tavily_results": search_results, "wikipedia_results": wikipedia_results}

def prepare_output(state: WebSearchState) -> WebSearchState:
    """
    Prepare output for user
    Input: state.search_results
    Output: state.response with formatted answer
    """
    tavily_results = state["tavily_results"]
    wikipedia_results = state["wikipedia_results"]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    structured_llm = llm.with_structured_output(Output)
    
    system_prompt = output_instructions.format(tavily_results=tavily_results, wikipedia_results=wikipedia_results)
    
    result = structured_llm.invoke([SystemMessage(content=system_prompt),
                            HumanMessage(content="Prepare the results from the given sources,include references to the sources")])    
    
    final_answer = result["final_answer"]
    references = result["references"]

    return {"final_answer": final_answer, "references": references}
    
    
def evaluate_results(state: WebSearchState) -> WebSearchState:
    """
    Evaluate search results for relevance and completeness
    Input: state.search_results
    Output: state.evaluation, state.requires_refinement, state.refinement_strategy
    """
    
    tavily_results = state["tavily_results"]
    wikipedia_results = state["wikipedia_results"]
    
    search_results_answer = tavily_results[0]['answer']
    initial_query = state["initial_query"]
    
    structured_llm = llm.with_structured_output(Evaluation)
    
    system_prompt = evaluation_instructions.format(answer=search_results_answer, question=initial_query)
    
    result = structured_llm.invoke([SystemMessage(content=system_prompt),
                                HumanMessage(content="Based on the above instructions, please evaluate the search results and determine if the answer covers the question.")])    
    print(result)
    return {"evaluation": result}

        
def route_after_evaluation(state: WebSearchState) -> Dict[str, str]:
    """
    Determine whether to refine search or prepare output
    Input: state.requires_refinement
    Output: Route to "web_search_agent" or END
    """
    
    if state["evaluation"]["is_refinement_needed"]:
        return "web_search_agent"
    else:
        return END
 
# Retrieval subgraph setup
retrieval_builder = StateGraph(WebSearchState)
retrieval_builder.add_node("web_search_agent", web_search_agent)
retrieval_builder.add_node("evaluate_results", evaluate_results)
retrieval_builder.add_node("prepare_output", prepare_output)

# Retrieval subgraph edges
retrieval_builder.add_edge(START, "web_search_agent")
retrieval_builder.add_edge("web_search_agent","prepare_output")
retrieval_builder.add_edge("prepare_output","evaluate_results")
retrieval_builder.add_conditional_edges("evaluate_results", route_after_evaluation, ["web_search_agent", END])


# Main graph setup
builder = StateGraph(OverallState)
builder.add_node("check_ambiguity", check_ambiguity)
builder.add_node("analyze_query", analyze_query)
builder.add_node("human_feedback", human_feedback)

company_information_retrieval = retrieval_builder.compile()
builder.add_node("company_information_retrieval", company_information_retrieval)
builder.add_node("generate_response", generate_response)

# Main graph edges
builder.add_edge(START, "check_ambiguity")
builder.add_conditional_edges("check_ambiguity", route_after_ambiguity, ["human_feedback", "analyze_query"])
builder.add_edge("analyze_query", "company_information_retrieval")
builder.add_edge("human_feedback", "check_ambiguity")
builder.add_edge("company_information_retrieval", "generate_response")
builder.add_edge("generate_response", END)



memory = MemorySaver()
config_specs = memory.get_config_specs() if hasattr(memory, 'get_config_specs') else None

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_feedback"]
)

from IPython.display import Image, display
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

# Input

user_query = "wHO İS THE CEO OF OPENAI"
thread = {"configurable": {"thread_id": "1"}}

async def run_graph():
    async for event in graph.astream({"initial_query": user_query}, thread, stream_mode="updates"):
        print(event)


# Call the async function using asyncio.run()
asyncio.run(run_graph())


