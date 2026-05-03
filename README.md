![build-and-test-python](https://github.com/Alex-Glebov/ZigZag/actions/workflows/build-and-test-python.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/zigzag-dt.svg)](https://badge.fury.io/py/zigzag-dt)
[![GitHub stars](https://img.shields.io/github/stars/Alex-Glebov/ZigZag)](https://github.com/Alex-Glebov/ZigZag/stargazers)
[![GitHub license](https://img.shields.io/github/license/Alex-Glebov/ZigZag)](https://github.com/Alex-Glebov/ZigZag/blob/main/LICENSE.txt)
![PyPI - Downloads](https://img.shields.io/pypi/dm/zigzag-dt)


# zigzag-dt

`zigzag-dt` provides functions for identifying the peaks and valleys of a time
series. Additionally, it provides a function for computing the maximum drawdown.

Install with pip:

```bash
pip install zigzag-dt
```

For fastest understanding, [view the IPython notebook demo tutorial](https://github.com/Alex-Glebov/ZigZag/blob/master/zigzag_demo.ipynb).

## API Aliases

To avoid name collisions with other zigzag packages, all main functions are also
exposed with a `zz_` prefix:

```python
from zigzag_dt import zz_pivots, zz_line, zz_max_drawdown
```

## Contributing

This is an admittedly small project. Still, if you have any contributions,
please [fork this project on github](https://github.com/Alex-Glebov/ZigZag) and
send me a pull request.
