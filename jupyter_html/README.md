# Conversion of Jupyter notebooks to html
Managing notebooks and analysis using version control

Notebooks can be:
- stored as `.py` files instead of `.ipynb` automatically using [jupytext](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html)
- parametrised and executed using [papermill](https://papermill.readthedocs.io/en/latest/)
- converted to `.html` files and served using Gitlab pages


## Getting started

Using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```bash
conda env create --name notebook-workflow-template -f environment.yaml --force
conda activate notebook-workflow-template
```

Using [pyenv](https://github.com/pyenv/pyenv), first install conda:

```bash
pyenv install miniconda3-4.3.30
pyenv local miniconda3-4.3.30
```

Then install python dependencies:

```bash
conda env create --name notebook-workflow-template -f environment.yaml --force
pyenv local miniconda3-4.3.30/envs/notebook-workflow-template
```

Removing a pyenv env:

```bash
rm -rf ~/.pyenv/versions/miniconda3-4.3.30/envs/notebook-workflow-template
pyenv local miniconda3-4.3.30
```

## Usage

- create notebooks and store them as .py files in `templates/`
- execute them using papermill and store the outputs in `generated/`
- convert them to `.html` and store them in `public/` where they can be viewed on Gitlab pages.

### Notebook development

1. Create a notebook in the `templates/` folder, or pull latest changes from master
2. Ensure notebooks are paired to `py` format using jupytext
   ```bash
   jupytext --set-formats ipynb.py --sync templates/*.ipynb
   ```
3. Modify the notebook either using jupyter server
  ```bash
  jupyter notebook templates/xyz.ipynb
  ```
4. Or parametrise and run it using papermill
  ```bash
  papermill templates/0.1-example.ipynb generated/0.1-example.ipynb -p N_DAYS 3
  ```
5. When your notebook is modified, the corresponding `py` file will be updated automatically
6. Stage and commit the `py` files in the `templates/` folder

### Publishing to Gitlab pages

1. Convert `ipynb` to `html` using [nbconvert](https://github.com/jupyter/nbconvert)
  ```bash
  jupyter nbconvert --to html generated/0.1-example.ipynb --no-input --template classic --output ../public/0.1-example.html
  ```
2. Stage and commit the `html` files in the `public/` folder
3. Update the index of the Gitlab pages document
  ```bash
  tree public -H '.' -L 3 --noreport --charset utf-8 > public/index.html
  ```
4. Stage and commit the changes in `public/`
5. Verify that the `pages` CI/CD stage runs successfully
