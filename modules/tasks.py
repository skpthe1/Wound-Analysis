from crewai import Task

def create_analysis_task(data_analysis_agent, data):
    return Task(
        description=f"""
        Analyze the following wound care product sales data: {data}. Your analysis must be very detailed and comprehensive.

        Specifically:

        1.  Identify key weekly sales trends, noting any significant increases or decreases in sales volume. Present this as a percentage change.
        2.  Assess the performance of each wound healing product in the dataset. Determine which products are consistently top performers and which are underperforming.
        3.  Extract key statistics, such as:
            *   Total sales for each product
            *   Average weekly sales
            *   Maximum and minimum weekly sales
        4.  Identify any correlations between product sales and specific weeks or time periods. Are there any seasonal trends or patterns?
        5.  Provide a summary of the most significant insights from your analysis, highlighting the most important trends and patterns in the data.

        Present your findings in a clear and concise manner, suitable for use in generating hypotheses.
        """,
        agent=data_analysis_agent,
    )


def create_hypothesis_task(hypothesis_agent, analysis_results):
    return Task(
        description=f"""
        Based on the following analysis results: {analysis_results}, generate a set of hypotheses that could explain the observed trends and patterns in wound care product sales. Your hypotheses should be creative, insightful, and testable.

        Consider the following:

        1.  What potential factors could be driving the observed weekly sales trends?
        2.  What could be causing some products to outperform others?
        3.  Are there any external factors (e.g., market conditions, competitor activities) that might be influencing sales?
        4.  Can you suggest any actionable strategies for improving sales or product performance based on your hypotheses?
        5.  Come up with at least 3 distinct hypotheses.

        Each hypothesis should be clearly stated and include a brief rationale for why you believe it to be plausible.
        """,
        agent=hypothesis_agent,
    )


def create_validation_task(validation_agent, hypotheses):
    return Task(
        description=f"""
        Validate the following hypotheses: {hypotheses}. For each hypothesis, assess its validity based on available data and provide a confidence score (out of 100) that reflects your level of certainty.

        Consider the following:

        1.  Is there any direct evidence in the data to support or refute the hypothesis?
        2.  Are there any potential confounding factors that could be influencing the results?
        3.  What are the limitations of the data, and how might these limitations affect the validity of the hypotheses?
        4.  Are there any additional data sources or analyses that could be used to further validate the hypotheses?

        Provide a detailed explanation for your confidence score, including a summary of the evidence you considered and any caveats or limitations.
        """,
        agent=validation_agent,
    )
