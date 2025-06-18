from setuptools import find_packages, setup

setup(
    name="311-ml-food-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "jupyter>=1.0.0",
        "setuptools>=58.0.0",
        "python-dotenv>=0.19.0",
    ],
    author="Aamid",
    description="A machine learning project for food prediction using student survey data",
    python_requires=">=3.8",
) 