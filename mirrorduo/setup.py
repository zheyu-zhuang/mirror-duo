# setup.py
from setuptools import setup, find_namespace_packages

setup(
    name="mirrorduo",
    version="0.1.0",
    description="Reflection-consistent visuomotor learning via mirroring",
    long_description_content_type="text/markdown",
    author="Zheyu Zhuang et al.",
    url="https://github.com/zheyu-zhuang/mirror-duo",
    packages=find_namespace_packages(include=["mirrorduo*"], exclude=("tests", "docs", "examples")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "torch>=1.9",
        "einops>=0.4.1",
        "kornia",
    ],
    python_requires=">=3.8",
    license="MIT",
)