from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements
INSTALL_REQUIRES = [
    'openai>=1.0.0',
    'anthropic>=0.3.0',
    'google-generativeai>=0.1.0',
    'python-dotenv>=0.19.0',
    'typing-extensions>=4.0.0',
]

# Development requirements
DEV_REQUIRES = [
    'pytest>=6.0',
    'pytest-asyncio>=0.14.0',
    'pytest-cov>=2.0',
    'black>=22.0',
    'isort>=5.0',
]

setup(
    name="llm_dialog_manager",
    version="0.1.0",
    author="xihajun",
    author_email="work@2333.fun",
    description="A Python package for managing AI chat conversation history",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xihajun/ai-chat-history",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': DEV_REQUIRES,
        'test': [
            'pytest>=6.0',
            'pytest-asyncio>=0.14.0',
            'pytest-cov>=2.0',
        ],
        'lint': [
            'black>=22.0',
            'isort>=5.0',
        ],
        'all': DEV_REQUIRES,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ai-chat-history/issues",
        "Documentation": "https://github.com/yourusername/ai-chat-history#readme",
        "Source Code": "https://github.com/yourusername/ai-chat-history",
    },
) 