import sys

def error_message_detail(error, error_detail: sys):
    """
    Formats the error message to include the file name, line number, and error message.
    """
    # Gets the third element of the traceback (where the error occurred)
    _, _, exc_tb = error_detail.exc_info() 
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = (
        f"Error occurred in python script name: [{file_name}] "
        f"line number: [{line_number}] error message: [{str(error)}]"
    )
    
    return error_message

class CustomException(Exception):
    """
    Custom exception class to raise and log errors with detailed file and line information.
    """
    def __init__(self, error_message, error_detail: sys):
        # Passes the formatted message to the base Exception class
        super().__init__(error_message) 
        
        # Stores the detailed, traceable error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        
    def __str__(self):
        # This is what gets printed when the exception is raised
        return self.error_message