from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-subtitle-generator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful Python application that automatically generates synchronized subtitles from audio files using AI transcription",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-subtitle-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
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
    entry_points={
        "console_scripts": [
            "subtitle-generator=advanced_subtitle_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.py"],
    },
    keywords="subtitle generator whisper ai transcription video audio",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-subtitle-generator/issues",
        "Source": "https://github.com/yourusername/advanced-subtitle-generator",
        "Documentation": "https://github.com/yourusername/advanced-subtitle-generator#readme",
    },
) 