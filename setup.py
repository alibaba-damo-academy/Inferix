from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("requirements-torch.txt", "r", encoding="utf-8") as fh:
    torch_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="inferix",
    version="1.0.0",
    author="Inferix Team",
    author_email="contact@inferix.ai",
    description="Next-Generation Inference Engine for Immersive World Synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inferix/inferix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=torch_requirements + requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "profiling": ["pynvml", "psutil"],
    },
    entry_points={
        "console_scripts": [
            "inferix-self-forcing=inferix.pipeline.self_forcing.pipeline:main",
            "inferix-causvid=inferix.pipeline.causvid.pipeline:main",
            "inferix-magi=inferix.pipeline.magi.pipeline:main",
        ],
    },
)