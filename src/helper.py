from src.log_config import get_logger

import json

import requests
from tqdm import tqdm


logger = get_logger(__name__)

def process_download_streaming_response(response: requests.Response, model_name: str):
    '''
    Processes the streaming output from the downloading model from ollama api/pull
    
    Args:
        response (requests.Response): The request response
        model_name (str): Downloaded Model name
        
    Return:
        JSON: a parsed json 
    '''
    past_progress = -1
    past_status = None
    for line in response.iter_lines():
        if not line:
            continue
        try:
            status = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    
    
        if "total" in status:
            # with tqdm(total=status['total'], desc=f"Downloading {model_name} >> {status["status"]}") as pbar:
            if "status" in status and "completed" in status and "total" in status:
                if past_progress < int((status["completed"]/status["total"])*100):
                    logger.info(f"Downloading {status["status"]}\t>>\t{int((status["completed"]/status["total"])*100)}% completed." )
                    past_progress = int((status["completed"]/status["total"])*100)

            if past_status != status["status"]:
                logger.info(f"Downloading {status["status"]}" )
                past_progress = -1
                past_status = status["status"]
                    
                    
            if status.get("status") == "success":
                logger.info(f"Model {model_name} downloaded successfully")
                return {"message": "Model has been downloaded"}
