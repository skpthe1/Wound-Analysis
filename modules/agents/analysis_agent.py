from langchain_openai import AzureChatOpenAI
from crewai import Agent


def create_analysis_agent(api_key, azure_endpoint, deployment_name, api_version ):
    return Agent(
        role="Data Analyst",
        goal="Analyze wound data trends and identify key insights.",
        backstory="A data analyst with expertise in healthcare, specializing in wound care products and sales data.",
        verbose=True,
        llm=AzureChatOpenAI(        
            api_key=api_key,
            openai_api_version=api_version,  # Replace with your API version
            azure_deployment=deployment_name,
            azure_endpoint=azure_endpoint,
        ),
    )
