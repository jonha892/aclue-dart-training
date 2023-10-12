#!/bin/sh
#
# Pre-commit hook for clearing output cells from commited analysis Jupyter notebooks.
#

# Check if Jupyter is accessible
if ! command -v jupyter &> /dev/null; then
    echo "Error: no jupyter in PATH, check if env is activated"
    echo "commit interrupted, please try again after fixing the problem"
    exit 1
fi

echo "Running pre-commit hook to clear output from *.ipynb notebooks."
for notebook in $(git diff --cached --name-only | grep -E '\.ipynb$')
do
	echo "Clearing output from $notebook"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearOutputPreprocessor.remove_metadata_fields=[] --to notebook --inplace $notebook
	git add $notebook
done

exit 0