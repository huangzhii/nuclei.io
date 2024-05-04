from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="nuclei.io",
    version="1.0",
    description="",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=["software"],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)