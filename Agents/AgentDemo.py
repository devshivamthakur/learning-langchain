from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
load_dotenv()
import os

WHETHER_API_KEY = os.getenv('WHETHER_API_KEY')


#1 setup llm
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Replace with any HF LLM you want
    task="text-generation",
    max_new_tokens=150,
    temperature=0.5
)

llm = ChatHuggingFace(llm=llm_endpoint)

#2 setup tools
search_tool = DuckDuckGoSearchRun()

#create whether tool
@tool
def get_Whether_data(city:str):
    """This function fetched the current whether data for a given city"""

    url = f"https://api.weatherapi.com/v1/current.json?q={city}&key={WHETHER_API_KEY}"
    response = requests.get(url)
    return response.json()


tools = [search_tool, get_Whether_data]
# Step 3: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 4: Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Step 5: Create the AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Step 6: Run the agent with a sample query
query = "What is the current weather in Dindori(Madhya pradesh) and who is the MLA this city?"
response = agent_executor.invoke({
    "input": query
})

print(response)
