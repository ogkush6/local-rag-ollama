import ollama
from tqdm import tqdm





def __pull_model(name: str) -> None:
    current_digest, bars = "", {}
    for progress in ollama.pull(name, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            print(progress.get("status"))
            continue

        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(
                total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True
            )

        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest


def __is_model_available_locally(model_name: str) -> bool:
    try:
        ollama.show(model_name)
        return True
    except ollama.ResponseError as e:
        return False


def check_if_model_is_available(model_name, host="localhost"):
    """
    Check if a specific model is available in Ollama.
    
    Args:
        model_name (str): Name of the model to check
        host (str): Hostname or IP address of the Ollama server
        
    Returns:
        bool: True if available, raises Exception if not
    """
    try:
        # Set the base URL for the ollama client
        client = ollama.Client(host=f"http://{host}:11434")
        
        # Get list of available models
        models = client.list()
        
        # Convert to list of model names
        model_names = [model['name'] for model in models['models']] if 'models' in models else []
        
        # Check if our model is in the list
        if model_name in model_names:
            return True
        else:
            # Try to pull the model
            print(f"Model {model_name} not found. Attempting to pull...")
            for progress in client.pull(model_name, stream=True):
                if 'completed' in progress and progress['completed']:
                    print(f"\nModel {model_name} pulled successfully")
                    return True
                elif 'status' in progress:
                    print(f"\r{progress['status']}", end="")
    except Exception as e:
        raise Exception(f"Error connecting to Ollama at {host}: {str(e)}")
    
    raise Exception(f"Model {model_name} is not available and could not be pulled")


