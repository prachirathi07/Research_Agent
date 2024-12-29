from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API keys
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# Configure Groq model
groq_model = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-groq-70b-8192-tool-use-preview"
)

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Research agent
research_agent = Agent(
    name="Research Agent",
    role="NYT Senior Researcher",
    model=groq_model,
    tools=[DuckDuckGo(), Newspaper4k()],
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        "For a given topic, search for the top 5 links.",
        "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
        "Analyse and prepare an NYT worthy article based on the information.",
    ],
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Multi AI agent
multi_ai_agent = Agent(
    name="Research Team",
    team=[web_search_agent, research_agent],
    instructions=[
        "Use web search agent for initial information gathering",
        "Use research agent for detailed analysis and article writing",
        "Always include sources"
    ],
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    # Test the research agent
    print("\nTesting Research Agent:")
    research_agent.print_response("Simulation theory", stream=True)
    
    print("\nTesting Multi AI Agent:")
    multi_ai_agent.print_response("Current developments in quantum computing", stream=True)