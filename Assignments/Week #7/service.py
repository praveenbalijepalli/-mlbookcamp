
import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray

tag = "mlzoomcamp_homework:qtzdz3slg6mwwdu5"  # Bentoml sklearn model's tag

model_ref = bentoml.sklearn.get(tag)  # Reference object to Bentoml sklearn model

model_runner = model_ref.to_runner() # Run the Bentoml sklearn model using the reference object

svc = bentoml.Service("classifier", runners=[model_runner]) # Create service to run the Bentoml sklearn model 


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result  
