<!-- https://www.sphinx-doc.org/en/master/usage/installation.html -->
conda install sphinx

<!-- local build -->
<!-- https://www.sphinx-doc.org/en/master/usage/quickstart.html -->

sudo apt install make 

cd docs
make html

autodoc folder has the API import bits


docs/requirements.txt is the instructions for readthedocs. Should match (root)/requirements.txt other than the lines for sphinx.

https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst