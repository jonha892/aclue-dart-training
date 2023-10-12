# aclue-dart-training

This repo contains a code to train our machine learning model for detecting darts.

It uses an approach similar to SSD. However, we implemented the head, and loss function from scratch. Thus, our approach differs in some way `¯\_(ツ)_/¯`.


### pre-commit-hook

To clean the output of the notebook, copy the pre-commit-hook script into the git folder.

The pre-commit-hook will fail if jupyter isn't available.

`cp pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`