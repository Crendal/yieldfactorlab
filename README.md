# YieldFactorLab

A Python package for yield curve factor analysis using PCA and NSS models.

## Installation

```bash
pip install git+https://github.com/Crendal/yieldfactorlab.git
```

## Example

```python
from yieldfactorlab import YieldFactorLab

lab = YieldFactorLab(
    method="PCA",
    country="US",
    factors=3
)

result = lab.run()

print(result["summary"]["explained"])
```
