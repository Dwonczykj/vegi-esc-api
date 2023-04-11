from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

setup(
    name="vegi_esc_api",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Deployment of vegi esc server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vegiApp/vegi-esc-api",
    author="Joey Dwonczyk",
    author_email="joey@vegiapp.co.uk",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    # entry_points={"console_scripts": ["vegi_esc_api = nyc_taxi_fare.cli:train"]},
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "future",
        "flask",
        "pot",
        "python-dotenv",
        "aiohttp",
        "dataclasses",
        "boto3",
        "cachetools",
        "paramiko",
        "pymysql",
        "gensim",
        "gunicorn",
        "eventlet",
        "jsons",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "lightgbm",
        "pandas",
        "pyarrow",
        "uvicorn",
        "nltk",
        "pyyaml",
    ],
)