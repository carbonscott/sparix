import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparix",
    version="23.08.27",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Spherical Phi Auto-Regressive X-ray (SPARX)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peaknet",
    keywords = ['X-ray', 'Transformer', 'Auto-regressive generation'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
