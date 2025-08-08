from setuptools import setup, find_packages

setup(
    name="risksimpro",
    version="0.1.0",
    description="Advanced Monte Carlo Portfolio Risk Simulation Framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pymc3",
        "plotly",
        "dash",
        "qiskit",
        "arviz",
        "arch"
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'risksimpro=risksimpro.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
