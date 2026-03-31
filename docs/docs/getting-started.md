# Getting Started

Welcome to hiveframe! This guide will help you get up and running quickly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install hiveframe
```

## Quick Example

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta", "bandung"]})
df["city"] = ["DKI Jakarta", "West Java"]
print(df.head())
```

For more advanced usage, see the API Reference and Guides sections.
