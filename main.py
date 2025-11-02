import sys
from EAP.pipeline.train_pipeline import TrainingPipeline
from EAP.exception.exception import CustomException

if __name__ == '__main__':
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        
    except CustomException as e:
        print(e)
        
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")