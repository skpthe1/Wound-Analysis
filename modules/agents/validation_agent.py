from crewai import Agent
from langchain_openai import AzureChatOpenAI

def create_validation_agent(api_key, azure_endpoint, deployment_name, api_version):
    return Agent(
        role="Clinical Validator",
        goal="Validate medical hypotheses against clinical evidence",
        backstory="Board-certified wound care specialist with expertise in evidence-based medicine",
        verbose=True,
        llm=AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature= 0.1,
        ),
        max_iter=20,
        memory=False,
        tools=[]
    )

