from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Pydantic model representing the standardized API response for spoofing detection.
    
    Attributes:
        prediction (str): The classification result, typically 'real' or 'fake'.
        confidence (float): The confidence score of the prediction, 
                        represented as a percentage (0.0 to 100.0).
    """
    prediction: str
    confidence: float