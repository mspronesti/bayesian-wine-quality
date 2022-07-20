class NotFittedError(Exception):
    """Exception class for not fitted instances"""
    def __init__(self, message=""):
        super().__init__(message)
