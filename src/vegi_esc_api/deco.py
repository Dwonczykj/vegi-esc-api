import time
# ~ https://python.plainenglish.io/five-python-wrappers-that-can-reduce-your-code-by-half-af775feb1d5


def timer(func):
    def wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # return the result of the decorated function execution
        return result
    # return reference to the wrapper function
    return wrapper


def debug(func):
    '''An additional useful wrapper function can be created to facilitate debugging by printing the inputs and outputs of each function. '''
    def wrapper(*args, **kwargs):
        # print the fucntion name and arguments
        print(f"Calling {func.__name__} with args: {args} kwargs: {kwargs}")
        # call the function
        result = func(*args, **kwargs)
        # print the results
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper


def exception_handler(func):
    '''The exception_handler the wrapper will catch any exceptions raised within the divide function and handle them accordingly.c'''
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle the exception
            print(f"An exception occurred: {str(e)}")
            # Optionally, perform additional error handling or logging
            # Reraise the exception if needed
    return wrapper


def validate_input(*validations):
    '''4 — Input Validator
This wrapper function validates the input arguments of a function against specified conditions or data types. It can be used to ensure the correctness and consistency of the input data.

The other approach to do that is by creating countless assert lines inside the function we want for validating the input data.

To add validations to the decoration, we need to wrap the decorator function with another function that takes in one or more validation functions as arguments. These validation functions are responsible for checking if the input values meet certain criteria or conditions.

The validate_input function itself acts as a decorator now. Inside the wrapper function, the input and the keyword arguments are checked against the provided validation functions. If any argument fails the validation, it raises a ValueError with a message indicating the invalid argument.'''
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i, val in enumerate(args):
                if i < len(validations):
                    if not validations[i](val):
                        raise ValueError(f"Invalid argument: {val}")
            for key, val in kwargs.items():
                if key in validations[len(args):]:
                    if not validations[len(args):][key](val):
                        raise ValueError(f"Invalid argument: {key}={val}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(max_attempts, delay=1):
    '''5 — Retry
This wrapper retries the execution of a function a specified number of times with a delay between retries. It can be useful when dealing with network or API calls that may occasionally fail due to temporary issues.

To implement that we can define another wrapper function to our decorator, similar to our previous example. However this time rather than providing validation functions as input variables we can pass specific parameters such as the max_attemps and the delay .

When the decorated function is called, the wrapper function is invoked. It keeps track of the number of attempts made (starting at 0) and enters a while loop. The loop attempts to execute the decorated function and immediately returns the result if successful. However, if an exception occurs, it increments the attempts counter and prints an error message indicating the attempt number and the specific exception that occurred. It then waits for the specified delay using time.sleep before attempting the function again.'''
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)
            print(f"Function failed after {max_attempts} attempts")
        return wrapper
    return decorator
