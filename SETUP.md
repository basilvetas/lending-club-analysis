# Environment Setup [Optional]

### Pipenv

If you don't have Pipenv installed you can install it on Mac with:

	brew install pipenv

If you're not familiar with Pipenv, you can learn about it [here](https://pipenv.readthedocs.io/en/latest/).

To start, in your project folder run:

	pipenv --python 3.6.5
	pipenv run pip install pip==18.0
	pipenv install

This will create a Pipfile and Pipfile.lock. Please add any python packages you use to this Pipfile. To add packages to the Pipfile run:

	pipenv install <PACKAGE-NAME>

Next run:

	pipenv shell

This will bring up a terminal in your virtualenv like this:

	(my-virtualenv-name) bash-4.4$

### Jupyter Notebook

Run:

	pipenv install jupyter

To create a Jupyter Notebook kernel for this virtual env, in the shell run:

	python -m ipykernel install --user --name=`basename $VIRTUAL_ENV` --display-name "My Virtualenv Name"

Launch jupyter notebook within the shell:

	jupyter notebook

When running an existing notebook, make sure the kernel "My Virtualenv Name" is selected (if not, you should be able to select it as an option in Kernel -> Change Kernel).

When creating a new notebook, make sure to select "My Virtualenv Name" as the default kernel.

In the future when opening the project, you should only have to run:

	pipenv shell
	jupyter notebook

### Authors
* [Basil Vetas](http://github.com/basilvetas)