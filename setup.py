from setuptools import find_packages, setup

setup(
    name='analytix',
    version='0.1',
    author='Cyril Joly',
    description='A Python package for data analysis and model optimization.',
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
