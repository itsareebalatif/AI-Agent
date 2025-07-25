from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os

# Load environment variables (GROQ_API_KEY, ANTHROPIC_API_KEY)
load_dotenv()
llm = ChatGroq(model="llama3-70b-8192") 

# Define expected response schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Pydantic parser for validating and formatting output
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a research assistant that will help generate a research paper.\n"
         "Answer the user query and use necessary tools.\n"
         "Wrap the output in this format and provide no other text:\n{format_instructions}"),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# LLM setup

# Optional: Replace or try Groq LLM
# llm = ChatGroq(model_name="llama3-70b-8192")

# Agent setup (no tools used yet)
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=[])

# AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

# Ask user
query = input("🔍 What can I help you research? ")

# Invoke agent
raw_response = agent_executor.invoke({"query": query})

# Parse the response
try:
    structured_response = parser.parse(raw_response["output"])
    print("\n✅ Parsed Response:")
    print(structured_response)
except Exception as e:
    print("❌ Error parsing response:", e)
    print("🔁 Raw Response:", raw_response)
