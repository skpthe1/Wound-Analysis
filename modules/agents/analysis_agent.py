from crewai import Agent
from langchain_openai import AzureChatOpenAI

def create_analysis_agent(api_key, azure_endpoint, deployment_name, api_version):
    return Agent(
        role="Clinical Data Analyst",
        goal="Analyze wound healing trends and product performance",
        backstory="Expert in statistical analysis of medical data with 10+ years experience in wound care",
        verbose=True,
        llm=AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=deployment_name,
            model_kwargs={"response_format": {"type": "json_object"}}
        ),
        memory=True
    )
