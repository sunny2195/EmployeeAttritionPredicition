import sys
from EAP.pipeline.train_pipeline import TrainingPipeline
from EAP.exception.exception import CustomException

if __name__ == '__main__':
    try:
        # Create an instance of the Training Pipeline
        pipeline = TrainingPipeline()
        
        # Run the full pipeline
        pipeline.run_pipeline()
        
    except CustomException as e:
        # Print the detailed custom error message
        print(e)
        
    except Exception as e:
        # Catch any root-level errors and display them
        print(f"An unexpected error occurred in the main execution: {e}")