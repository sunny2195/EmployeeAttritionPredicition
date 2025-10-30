from setuptools import setup, find_packages

setup(
    name="EAP",
    version="0.0.1",
    author="Your Name",
    description="MLOps Pipeline for Employee Attrition Prediction.",
    # This finds the EAP folder and turns it into a package
    # Crucial for Docker to run 'pip install .'
    packages=find_packages(), 
    install_requires=[] # Dependencies are handled by requirements.txt
)