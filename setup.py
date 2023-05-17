import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setuptools.setup(
    name='replica',
    version='0.0.1',
    author='Jacopo Venturin, Clark Templeton',
    description='Package to perform parallel tempering simulation based on openmm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jacopoventurin/replica1.git',
    license='MIT',
    packages=['replica'],
    install_requires=install_requires,
)
