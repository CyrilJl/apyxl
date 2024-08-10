from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='analytix',
    version='0.1',
    author='Cyril Joly',
    description='A Python package for data analysis and model optimization.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xgboost',
        'scikit-learn',
        'shap',
        'hyperopt',
        'matplotlib'
    ],
)
