# How to release

## Setup

Install required packages:
```
$ conda env create -f binder/environment.yml
```

Or if needed, update your conda environment:
```
$ conda env update -f binder/environment.yml
```

Install locally the Python package under development:
```
$ bash binder/postBuild
```

For Zenodo integration, see [Making Your Code Citable](https://guides.github.com/activities/citable-code/).

To publish a package in [PyPI](https://pypi.org/), create an [account](https://pypi.org/account/register/) first.


## Tests

Before any release, double-check all tests had run successfully:
```
$ make tests
```

