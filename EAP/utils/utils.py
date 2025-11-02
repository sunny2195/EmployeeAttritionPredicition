import yaml
from pathlib import Path
from EAP.exception.exception import CustomException 
import sys
import os
import pickle

def read_yaml(path_to_yaml: Path) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    Used for reading schema.yaml and any future parameters.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            print(f"YAML file loaded successfully from: {path_to_yaml}")
            return content
    except FileNotFoundError:
        raise CustomException(f"YAML file not found at: {path_to_yaml}", sys)
    except yaml.YAMLError as e:
        raise CustomException(f"Error parsing YAML file: {e}", sys)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: Path, obj):
    """
    Saves a Python object (like a trained model or scaler) to a binary file (.pkl).
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        print(f"Object successfully saved to: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: Path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except FileNotFoundError:
        raise CustomException(f"Object file not found at: {file_path}", sys)
    except Exception as e:
        raise CustomException(e, sys)