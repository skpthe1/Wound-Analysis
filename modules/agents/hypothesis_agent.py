from crewai import Agent
from langchain_openai import AzureChatOpenAI

def create_hypothesis_agent(api_key, azure_endpoint, deployment_name, api_version):
    return Agent(
        role="Medical Hypothesizer",
        goal="Generate clinically relevant hypotheses from data patterns",
        backstory="Senior researcher specializing in deriving testable medical hypotheses from complex datasets",
        verbose=True,
        llm=AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0.1,
        ),
        max_iter=20,
        memory=False,
        tools=[]  # Explicitly disable tools to avoid action expectations
    )
