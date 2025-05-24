"""
Setup script for the epitope classifier package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="epitope-classifier",
    version="0.1.0",
    author="Tariq Alagha",
    author_email="your.email@example.com",
    description="Machine learning models for identifying T-cell epitopes in cancer antigens",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/epitope-classifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "epitope-predict-binding=scripts.predict_binding_affinity:main",
            "epitope-visualize=scripts.visualize_predictions:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="bioinformatics machine-learning epitope immunotherapy cancer",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/epitope-classifier/issues",
        "Source": "https://github.com/yourusername/epitope-classifier",
        "Documentation": "https://epitope-classifier.readthedocs.io/",
    },
)