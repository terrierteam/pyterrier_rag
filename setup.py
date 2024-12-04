from setuptools import find_packages, setup


def get_version(path):
    for line in open(path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError(f"Unable to find __version__ in {path}")


def get_requirements(path):
    res = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if line:
            res.append(line)
    return res


setup(
    name='pyterrier-rag',
    version=get_version('pyterrier_rag/__init__.py'),
    author='Craig Macdonald',
    author_email='craig.macdonald@glasgow.ac.uk',
    description="PyTerrier RAG pipelines",
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/cmacdonald/pyterrier-rag',
    packages=find_packages(),
    include_package_data=True,
    entry_points={},
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
)
