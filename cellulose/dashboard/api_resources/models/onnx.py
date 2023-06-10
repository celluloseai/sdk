from pathlib import Path
import requests

BASE_URL = "http://localhost:8000"
#  BASE_URL = "https://dashboard.cellulose.ai"


def upload_onnx_model(api_key: str, onnx_file: Path) -> requests.Response:
    """
    This method uploads the ONNX model file to the Cellulose dashboard.

    Params
    ------
    api_key: str - The API key for the Cellulose dashboard.
    onnx_file: Path - The path to the ONNX model file.

    Returns
    -------
    requests.Response - The response from the Cellulose dashboard.
    """
    # Upload the ONNX model to the Cellulose dashboard.
    payload = {}
    files = [
        (
            "onnx_model_file",
            (
                onnx_file.name,
                open(onnx_file, "rb"),
                "application/octet-stream",
            ),
        )
    ]
    headers = {
        "X-API-Key": api_key,
    }

    response = requests.request(
        "POST",
        BASE_URL + "/v1/models/onnx/upload/sdk/",
        headers=headers,
        data=payload,
        files=files,
    )
    print(response.text)
    return response
