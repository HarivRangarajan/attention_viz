"""Setup script for attention_viz package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="attention-viz",
    version="0.1.0",
    author="Harivallabha Rangarajan, Aditya Shrivastava",
    author_email="harivallabha@example.com",
    description="A comprehensive toolkit for visualizing and inspecting attention patterns in transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarivRangarajan/attention_viz",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "matplotlib>=3.0.0",
        "plotly>=5.0.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "inspectus>=0.1.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "pre-commit>=2.15",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "attention-viz=attention_viz.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 