# LendingClub Analysis with Probablistic Programming

# Getting Started

To start, in your project folder run:
```bash
pipenv install --dev
pipenv shell
python -m ipykernel install --user --name=`basename $VIRTUAL_ENV` --display-name "Lending Club"
jupyter notebook
```

If you have issues with any of the commands above, see the `SETUP.md` file for more detailed instructions.

# Utilities

To lint (from within the virtualenv shell):
```bash
pytest --flake8
```

To clear all cache files (from within the virtualenv shell):
```bash
python final-project/utils/utils.py -c
```
or specify specific file types:

```bash
python final-project/utils/utils.py -c hdf pkl
```

# Final Project Instructions

The focus of this course is the final project. The goal is for you to choose a real world problem and to loop through the probabilistic modeling cycle using probabilistic programming. You will be expected to write, document, and report your analysis and findings. This will involve a significant amount of programming. Based on the number of students taking the course for credit, you will work either in groups of two or three. I will provide some suggestions; however, you are encouraged to find a problem in a field that excites you.

You will produce an 8-page final report and present your findings to the class in a short presentation at the end of the term. This project will measure your cumulative understanding of the material while providing you with a supportive environment to try out your new skills. Each student within a group will receive an individual grade, corresponding to their involvement in the project. [Source](http://www.proditus.com/syllabus2018.html)

More instructions [here](https://github.com/akucukelbir/probprog-finalproject)

# Box's Loop

![Box's Loop](final-project/resources/boxs-loop.png)

# References

Machine Learning: A Probabilistic Perspective, by Kevin P. Murphy, MIT Press, 2012