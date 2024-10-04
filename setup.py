from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='apyxl',
    version='0.1.4',
    author='Cyril Joly',
    description='A Python package for data analysis and model optimization.',
    url='https://github.com/CyrilJl/apyxl',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xgboost>=2.0.0',
        'scikit-learn',
        'shap',
        'hyperopt',
        'matplotlib'
    ],
)
