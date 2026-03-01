import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stream",
    description="System Code for reactor analysis using modular DAE coupling of components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=("tests*",)),
    package_data={"": ["standards/*.csv"]},
    version="1.1.0",
    python_requires=">=3.11",
)
