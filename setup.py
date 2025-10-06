from setuptools import setup, find_packages

setup(
    name="stock-predictor-pro",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'fastapi',
        'uvicorn',
    ]
)