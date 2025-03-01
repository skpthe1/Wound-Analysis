from crewai import Task
from typing import Dict, Any, List
import pandas as pd
from modules.analysis import summarize_data, summarize_analysis
import json
from crewai import Agent

def create_analysis_task(agent: Agent, data: pd.DataFrame) -> Task:
    """Create a task for analyzing wound care data."""
    summary = summarize_data(data)
    return Task(
        description=f"""Analyze the following wound care dataset and provide the results in json format:

{summary}

Include the following in the json output:
1. Weekly trends with visualizations.
2. Top-performing products by average wound area.
3. Correlation matrix between numerical variables.

Note: Ensure the response is a valid json object, and keep it as concise as possible to avoid length limits. Do not use Chain of thought. Do not add any extra text or explanations. Be as brief as humanly possible. Make sure to return a valid JSON object. This is not the last task, there are more steps after this.""",
        agent=agent,
        expected_output="A json object with weekly trends, product performance, and correlation matrix.",
        output_json=True,
        max_iter=20,  # Increased iteration limit
        max_rpm=15000,  # Increased RPM
    )

def create_hypothesis_task(agent: Agent, analysis_results: Dict[str, Any]) -> Task:
    """Create a task for generating hypotheses based on analysis results."""
    summary = summarize_analysis(analysis_results)
    truncated_summary = summary[:250] + "..." if len(summary) > 250 else summary  # Reduced the size to be sure
    return Task(
        description=f"""Using this summary:

{truncated_summary}

Generate exactly 2 hypotheses based on the patterns in the summary. Each hypothesis must be one sentence.

Return the two hypotheses as a string, separated by a new line.

Make sure to only provide the hypotheses. Do not add any other text.
Do not use chain of thought. Provide directly the result. This is not the last task, there are more steps after this.""",
        agent=agent,
        expected_output="Two hypotheses as a string, separated by a new line.",
        output_json=False,
        max_iter=20,  # Increased iteration limit
        max_rpm=15000,  # Increased RPM
    )

def create_validation_task(agent: Agent, hypotheses: List[str]) -> Task:
    """Create a task for validating hypotheses."""
    hypotheses_str = "\n".join([f"- {h}" for h in hypotheses])

    return Task(
        description=f"""Validate the following hypotheses:

{hypotheses_str}

For each hypothesis, provide a concise validation in the following format:
- Status: supported/unsupported/inconclusive
- Evidence: [a single, very short sentence]

Separate the validations for each hypothesis with a blank line.

Do not include any extra text or explanations. Be as brief as possible.
Do not use chain of thought. Provide directly the result.
Make sure to provide the status and the evidence in lowercase.

Example:
- Status: supported
- Evidence: data shows consistent trends

- Status: inconclusive
- Evidence: insufficient data for confirmation

This is the last task; make sure to give the final output.""",
        agent=agent,
        expected_output="Validation results for each hypothesis in plain text, with status and evidence.",
        output_json=False,
        max_iter=20,  # Increased iteration limit
        max_rpm=15000,  # Increased RPM
    )