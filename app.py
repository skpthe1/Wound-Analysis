import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Crew
from modules.data_loader import load_data
from modules.analysis import analyze_trends
from modules.agents.analysis_agent import create_analysis_agent
from modules.agents.hypothesis_agent import create_hypothesis_agent
from modules.agents.validation_agent import create_validation_agent
from modules.tasks import create_analysis_task, create_hypothesis_task, create_validation_task

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
  

def main():
    st.title("🏥 Wound Analysis AI Assistant")

    uploaded_file = st.file_uploader("Upload Wound Data CSV", type=["csv"])

    if uploaded_file:
        try:
            data = load_data(uploaded_file)
            st.session_state.data = data
            st.success("Data loaded successfully!")
        except ValueError as e:
            st.error(f"Error loading data: {str(e)}")
            return

    if st.button("🔍 Analyze Trends"):
        if 'data' not in st.session_state:
            st.warning("Please upload data first.")
            return

        analysis = analyze_trends(st.session_state.data)
        st.session_state.analysis = analysis

        st.subheader("Weekly Trends")
        st.plotly_chart(analysis['weekly_plotly'])

        st.subheader("Product Performance")
        st.plotly_chart(analysis['products_plotly'])

        st.subheader("Key Statistics")
        st.dataframe(analysis['weekly'])

    if st.button("💡 Generate Hypotheses"):
        if 'analysis' not in st.session_state:
            st.warning("Please analyze trends first.")
            return

        # Create the agents
        analysis_agent = create_analysis_agent(API_KEY, AZURE_ENDPOINT, DEPLOYMENT_NAME, OPENAI_API_VERSION)
        hypothesis_agent = create_hypothesis_agent(API_KEY, AZURE_ENDPOINT, DEPLOYMENT_NAME, OPENAI_API_VERSION)
        validation_agent = create_validation_agent(API_KEY, AZURE_ENDPOINT, DEPLOYMENT_NAME, OPENAI_API_VERSION)

        # Create the tasks and assign them to the agents
        analysis_task = create_analysis_task(analysis_agent, st.session_state.data)
        hypothesis_task = create_hypothesis_task(hypothesis_agent, st.session_state.analysis)
        validation_task = create_validation_task(validation_agent, "Generate hypotheses based on this analysis")

        # Create the crew
        crew = Crew(
            agents=[analysis_agent, hypothesis_agent, validation_agent],
            tasks=[analysis_task, hypothesis_task, validation_task],
            verbose=True,
        )

        # Kickoff the crew
        result = crew.kickoff()  # Run all task.

        st.session_state.hypotheses = result
        st.subheader("AI-Generated Hypotheses")
        st.write(result) # display the result of all tasks

    if st.button("✅ Validate Hypotheses"):
        if 'hypotheses' not in st.session_state:
            st.warning("Please generate hypotheses first.")
            return
        st.subheader("Validation Results")
        st.write(st.session_state.hypotheses)

if __name__ == "__main__":
    main()
