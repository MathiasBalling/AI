# Jupyter notebook from nvim

## Syntax

```python
# %% [markdown]
# This is a markdown cell

# %%
# This is a python cell
def f(x):
  return 3*x+1
```

## Usage

1. This makes a pair of synced py and ipynb files, `<name>.sync.py` and `<name>.sync.ipynb`.

   ```bash
   python -m jupyter_ascending.scripts.make_pair --base <name>
   ```

2. Start jupyter and open the notebook:

   ```bash
   python -m jupyter notebook <name>.sync.ipynb
   ```

3. Add some code to the `.sync.py` file, e.g.

   ```bash
     echo 'print("Hello World!")' >> <name>.sync.py
   ```

4. Sync the code into the jupyter notebook (without nvim plugin):

   ```bash
   python -m jupyter_ascending.requests.sync --filename <name>.sync.py
   ```

5. Run that cell of code (without nvim plugin):

   ```bash
   python -m jupyter_ascending.requests.execute --filename <name>.sync.py --line 16
   ```

## Installation

```bash
pip install jupyter_ascending && \
python -m jupyter nbextension     install jupyter_ascending --sys-prefix --py && \
python -m jupyter nbextension     enable jupyter_ascending --sys-prefix --py && \
python -m jupyter serverextension enable jupyter_ascending --sys-prefix --py
```

You can confirm it's installed by checking for `jupyter_ascending` in:

```bash
python -m jupyter nbextension     list
python -m jupyter serverextension list
```

If your jupyter setup includes multiple python kernels that you'd like to use with jupyter ascending, you'll need to complete this setup in each of those python environments separately.

Refer:
[jupyter_ascending/README.md at main · imbue-ai/jupyter_ascending](https://github.com/imbue-ai/jupyter_ascending/blob/main/README.md)
