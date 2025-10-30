import sys
import os

def error_message_detail(error, error_detail: sys):
    """
    Formats the error message to include the file name, line number, and error message.
    """
    # sys.exc_info() returns info about the current exception being handled
    _, _, exc_tb = error_detail.exc_info() 
    
    # Get the file name and line number where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = (
        f"Error occurred in python script name: [{file_name}] "
        f"line number: [{line_number}] error message: [{str(error)}]"
    )
    
    return error_message

class CustomException(Exception):
    """
    Custom exception class designed to raise and log errors with detailed 
    file and line information, replacing cryptic Python tracebacks.
    """
    def __init__(self, error_message, error_detail: sys):
        # Calls the parent Exception class constructor with the basic message
        super().__init__(error_message) 
        
        # Stores the detailed, traceable error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        
    def __str__(self):
        """Returns the fully detailed error message when the exception is printed."""
        return self.error_message
    
    def __repr__(self):
        """Provides a developer-friendly representation of the exception."""
        return f"CustomException('{self.error_message}')"

if __name__ == '__main__':
    # Example usage test
    try:
        a = 1/0
    except Exception as e:
        # Raising the custom exception for a ZeroDivisionError
        raise CustomException(e, sys)