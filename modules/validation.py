from pydantic import BaseModel, Field, validator
from typing import List, Dict

class Hypothesis(BaseModel):
    statement: str
    confidence: float = Field(ge=0, le=1)
    supporting_data: Dict[str, float]

    @validator('supporting_data')
    def check_supporting_data(cls, v):
        if not v:
            raise ValueError("Supporting data cannot be empty")
        return v

class ValidationResult(BaseModel):
    hypothesis: Hypothesis
    is_valid: bool
    explanation: str
    additional_evidence: List[str] = []

def validate_hypotheses(hypotheses: List[Hypothesis], data: Dict[str, float]) -> List[ValidationResult]:
    # This is a placeholder function. In a real-world scenario, 
    # this would involve more complex validation logic or potentially 
    # use the validation_agent to perform the validation.
    results = []
    for hypothesis in hypotheses:
        is_valid = all(data.get(key, 0) >= value for key, value in hypothesis.supporting_data.items())
        results.append(ValidationResult(
            hypothesis=hypothesis,
            is_valid=is_valid,
            explanation="Hypothesis is supported by the data" if is_valid else "Hypothesis is not fully supported by the data",
            additional_evidence=list(hypothesis.supporting_data.keys())
        ))
    return results
