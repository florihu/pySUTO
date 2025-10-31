from setuptools import setup, find_packages

setup(
    name="pySUTO",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "joblib",
        "tqdm",
        # add other dependencies here
    ],
)