from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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
    role="Search the web for the information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    add_history_to_messages=True,
    markdown=True
)

# Research agent 
research_agent = Agent(
    name="Research Agent",
    role="NYT Senior Researcher",
    model=groq_model,
    tools=[DuckDuckGo()],
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        "For a given topic, search for the top 5 links.",
        "Analyze and prepare an NYT-worthy article based on the information.",
    ],
    markdown=True,
    show_tool_calls=True,
    add_history_to_messages=True,
    add_datetime_to_instructions=True
)

# Create the playground app
app = Playground(agents=[research_agent,web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
