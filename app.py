import streamlit as st
import os
import pandas as pd
import json
import re
from dotenv import load_dotenv
from modules.data_loader import load_data
from modules.analysis import analyze_trends, summarize_data, summarize_analysis
from modules.agents import (
    create_analysis_agent,
    create_hypothesis_agent,
    create_validation_agent
)
from modules.tasks import (
    create_analysis_task,
    create_hypothesis_task,
    create_validation_task
)
from crewai import Crew
from plotly.io import from_json
from modules.validation import Hypothesis, ValidationResult
import traceback

load_dotenv()

def parse_hypothesis_output(output):
    try:
        if isinstance(output, str):
            text = output.strip()
        elif hasattr(output, 'raw'):
            text = output.raw.strip()
        else:
            raise ValueError("Unexpected output format from agent")
        
        # Clean up unwanted formatting (e.g., code blocks)
        text = re.sub(r'```(?:python)?\s*|\s*```', '', text).strip()
        
        # Split into sentences, allowing incomplete outputs
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        # Filter out empty or invalid sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5 and not s.strip().startswith('Agent stopped')]
        
        # Accept one or more hypotheses
        if not sentences:
            raise ValueError(f"No valid hypotheses found. Raw output: {output}")
        
        hypotheses = [Hypothesis(statement=sentence) for sentence in sentences]
        return hypotheses
    except ValueError as e:
        raise ValueError(f"Error parsing hypotheses: {str(e)}")

def parse_validation_output(output):
    try:
        if isinstance(output, str):
            text = output.strip()
        elif hasattr(output, 'raw'):
            text = output.raw.strip()
        else:
            raise ValueError("Unexpected output format from agent")
        blocks = re.split(r'\n\s*\n', text)
        validations = []
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                status_line = lines[0].strip()
                evidence_line = lines[1].strip()
                if status_line.startswith('- Status:') and evidence_line.startswith('- Evidence:'):
                    status = status_line.split(':')[1].strip()
                    evidence = evidence_line.split(':')[1].strip()
                    validations.append({"status": status, "evidence": evidence})
                else:
                    raise ValueError(f"Invalid format in validation block: {block}")
            else:
                raise ValueError(f"Insufficient lines in validation block: {block}")
        # Allow flexible number of validations
        if not validations:
            raise ValueError(f"No valid validations found. Raw output: {output}")
        return validations
    except ValueError as e:
        raise ValueError(f"Error parsing validation results: {str(e)}")

def main():
    st.title("🏥 Advanced Wound Care Analytics")

    # File uploader
    uploaded_file = st.file_uploader("Upload Wound Data CSV", type=["csv"])

    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.data = df
            st.success("✅ Data loaded successfully!")
            st.write(summarize_data(df))

            # Initialize API and agents once
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_ENDPOINT")
            deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
            api_version = os.getenv("OPENAI_API_VERSION")

            analysis_agent = create_analysis_agent(api_key, endpoint, deployment_name, api_version)
            hypothesis_agent = create_hypothesis_agent(api_key, endpoint, deployment_name, api_version)
            validation_agent = create_validation_agent(api_key, endpoint, deployment_name, api_version)

            # Analyze Data button at the top
            st.markdown("### Actions")
            analyze_clicked = st.button("🔍 Analyze Data")

            # Perform analysis action
            if analyze_clicked:
                with st.spinner("Analyzing data..."):
                    try:
                        analysis_results = analyze_trends(df)
                        st.session_state.analysis_results = analysis_results
                    except Exception as e:
                        st.error(f"🚨 Analysis failed: {str(e)}\n\nStack trace:\n{traceback.format_exc()}")

            # Display analysis results and Generate Hypotheses button
            st.markdown("---")
            if "analysis_results" in st.session_state:
                st.subheader("Analysis Summary")
                st.write(summarize_analysis(st.session_state.analysis_results))

                st.subheader("Weekly Trends")
                try:
                    weekly_fig = from_json(st.session_state.analysis_results['weekly_trends'])
                    st.plotly_chart(weekly_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render Weekly Trends chart: {str(e)}")

                st.subheader("Product Performance")
                try:
                    product_fig = from_json(st.session_state.analysis_results['product_performance'])
                    st.plotly_chart(product_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render Product Performance chart: {str(e)}")

                # Generate Hypotheses button after analysis results
                st.markdown("#### Next Step")
                hypothesis_clicked = st.button("💡 Generate Hypotheses")

                # Perform hypothesis generation action
                if hypothesis_clicked:
                    with st.spinner("Generating hypotheses..."):
                        try:
                            task = create_hypothesis_task(hypothesis_agent, st.session_state.analysis_results)
                            crew = Crew(agents=[hypothesis_agent], tasks=[task], verbose=True, max_iter=50)
                            result = crew.kickoff()
                            if result:
                                hypotheses = parse_hypothesis_output(result)
                                st.session_state.hypotheses = hypotheses
                            else:
                                st.error("Hypothesis generation returned no result")
                        except ValueError as e:
                            st.error(f"Error extracting hypotheses: {str(e)}. Raw output: {result}")
                        except Exception as e:
                            st.error(f"Error processing hypotheses: {str(e)}. Raw output: {result}")

            # Display hypotheses and Validate Hypotheses button
            if "hypotheses" in st.session_state:
                st.markdown("---")
                st.subheader("Generated Hypotheses")
                for i, hypothesis in enumerate(st.session_state.hypotheses, 1):
                    st.write(f"**Hypothesis {i}:**")
                    st.write(f"Statement: {hypothesis.statement}")

                # Validate Hypotheses button after hypothesis results
                st.markdown("#### Next Step")
                validate_clicked = st.button("✅ Validate Hypotheses")

                # Perform validation action
                if validate_clicked:
                    with st.spinner("Validating hypotheses..."):
                        try:
                            task = create_validation_task(validation_agent, [h.statement for h in st.session_state.hypotheses])
                            crew = Crew(agents=[validation_agent], tasks=[task], verbose=True, max_iter=50)
                            result = crew.kickoff()
                            if result:
                                if isinstance(result, str):
                                    validation_results = parse_validation_output(result)
                                elif hasattr(result, 'raw'):
                                    validation_results = parse_validation_output(result.raw)
                                else:
                                    raise ValueError(f"Unexpected result format: {type(result)}")
                                st.session_state.validation_results = validation_results
                            else:
                                st.error("Validation returned no result")
                        except ValueError as e:
                            st.error(f"Error extracting validation results: {str(e)}. Raw output: {result}")
                        except Exception as e:
                            st.error(f"Error processing validation: {str(e)}. Raw output: {result}")

            # Display validation results
            if "validation_results" in st.session_state and "hypotheses" in st.session_state:
                st.markdown("---")
                st.subheader("Validation Results")
                for i, (hypothesis, validation) in enumerate(zip(st.session_state.hypotheses, st.session_state.validation_results), 1):
                    st.write(f"**Validation {i}**")
                    st.write(f"Hypothesis: {hypothesis.statement}")
                    st.write(f"Status: {validation['status']}")
                    st.write(f"Evidence: {validation['evidence']}")

        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}\n\nStack trace:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()