from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements_reconocimiento.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nasa-astronaut-recognition",
    version="1.0.0",
    author="Antonio",
    author_email="antonio@nasa.gov",
    description="Real-time facial recognition and emotion analysis system for astronaut monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nasa-astronaut-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nasa-recognition=reconocimiento_antonio:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="nasa, astronaut, recognition, emotion, analysis, computer-vision, opencv, deepface, mediapipe",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nasa-astronaut-recognition/issues",
        "Source": "https://github.com/yourusername/nasa-astronaut-recognition",
        "Documentation": "https://github.com/yourusername/nasa-astronaut-recognition/wiki",
    },
)
