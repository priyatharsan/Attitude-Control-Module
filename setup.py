import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="attitudes-rconnorjohnstone",
    version="0.0.2",
    author="Connor Johnstone",
    author_email="connor@richardconnorjohnstone.com",
    description="A package for working with attitude descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rconnorjohnstone/Attitude-Control-Module",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
