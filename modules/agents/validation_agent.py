from langchain_openai import AzureChatOpenAI
from crewai import Agent

def create_validation_agent(api_key, azure_endpoint, deployment_name,api_version):
    return Agent(
        role="Hypothesis Validator",
        goal="Validate generated hypotheses using available data and provide a confidence score.",
        backstory="A detail-oriented analyst focused on verifying the accuracy and reliability of business hypotheses.",
        verbose=True,
        llm=AzureChatOpenAI(
            api_key=api_key,
            openai_api_version=api_version,  # Replace with your API version
            azure_deployment=deployment_name,
            azure_endpoint=azure_endpoint,
        ),
    )

