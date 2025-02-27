from crewai import Agent
from langchain_openai import AzureChatOpenAI 
from crewai import Agent

def create_hypothesis_agent(api_key, azure_endpoint, deployment_name,api_version):
    return Agent(
        role="Hypothesis Generator",
        goal="Generate insightful hypotheses based on data analysis to improve sales and product strategy.",
        backstory="A creative thinker skilled at identifying potential explanations for observed trends in business data.",
        verbose=True,
        llm=AzureChatOpenAI(
            api_key=api_key,
            openai_api_version=api_version,  # Replace with your API version
            azure_deployment=deployment_name,
            azure_endpoint=azure_endpoint,
        ),
    )
