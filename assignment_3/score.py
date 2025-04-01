import sklearn
import numpy as np
from typing import Tuple

def score(text: str, 
         model: sklearn.base.BaseEstimator,
         vectorizer: sklearn.feature_extraction.text.CountVectorizer,  # or TfidfVectorizer
         threshold: float) -> Tuple[bool, float]:
    X = vectorizer.transform([text])
    
    try:
        propensity = model.predict_proba(X)[0][1]
        print(propensity)
    except AttributeError:
        raise ValueError("Model must support predict_proba method")
    
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity 