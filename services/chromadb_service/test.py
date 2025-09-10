# test.py

import math
from typing import List

# -----------------------
# Standalone function
# -----------------------
def greet_user(name: str):
    """Print a greeting message"""
    print(f"Hello, {name}!")


# -----------------------
# Class with methods
# -----------------------
class User:
    """User class to handle authentication"""
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
    
    def login(self):
        """Simulate login"""
        if self.password == "1234":
            print(f"{self.username} logged in!")
        else:
            print("Invalid credentials")
    
    @staticmethod
    def validate_username(name):
        return isinstance(name, str) and len(name) > 0


# -----------------------
# Another class
# -----------------------
class Calculator:
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

    def factorial(self, n):
        """Recursive factorial"""
        if n == 0:
            return 1
        return n * self.factorial(n-1)


# -----------------------
# Function with decorator
# -----------------------
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Function is called")
        return func(*args, **kwargs)
    return wrapper

@decorator
def say_hello():
    print("Hello!")

# -----------------------
# One-liner function
# -----------------------
def square(x): return x * x
