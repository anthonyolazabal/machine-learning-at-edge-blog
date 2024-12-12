import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "qualitycheck:rwjjc7vxdc76lss6"

quality_check_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

quality_check_service = bentoml.Service(
    "quality_check", runners=[quality_check_runner])


@quality_check_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data: np.ndarray) -> np.ndarray:
    return quality_check_runner.predict.run(input_data)
