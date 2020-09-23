import setuptools
from pip.req import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

install_reqs = parse_requirements("requirements.txt", session="hack")

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setuptools.setup(
    name="example-pkg-blood",  # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=2.7",
    install_requires=reqs,
)

