import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="armacell",
    version="0.1.0",
    description="ARMA cell: a modular and effective approach for neural autoregressive modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phschiele/armacell",
    project_urls={
        "Bug Tracker": "https://github.com/phschiele/armacell/issues",
    },
    package_dir={"": "armacell"},
    packages=setuptools.find_packages(where="armacell"),
    python_requires=">=3.7",
    license='Apache License, Version 2.0',
    install_requires=[
        "tensorflow",
        "numpy",
        "statsmodels",  # Only used for comparison to classical ARMA models
        "matplotlib",  # Only used for plotting
    ],
)