# project-353

## CMPT 353 Data Science Project Summer 2020
We aim to answer the question of

    "How does walking pace differ between people? Does it vary by age, gender, and/or height?"

Through data collection/cleaning, statisical tests, and machine learning classifiers to see if our models can predict walking paces of slow, medium, or fast from the features stated above.

## Required Libraries
Numpy, Pandas, Seaborn, Matplotlib, Scipy, Sklearn and Statsmodels

## Instructions To Run The Code
(1) git clone the repository

(2) run these python files in order:

    python3 TJ-walk.py 
    python3 TJ-mom.py 
    python3 angus-walk.py 
    python3 angus-brother.py
    python3 kevin-walk.py
    python3 anna-walk.py
    python3 adam-walk.py
    
Outputs for this:

    Command Line:
        - step frequency, steps per min, time elapsed, steps taken, normal and levene test pvalues for all axes' acceleration
        
    Graphs:
        - one graph from TJ-walk.py displaying raw and filtered acceleration of all 3 axes, saved as "TJ-acceleration.png"
    
    CSV:
        - a csv file with that person's analyzed walk information appended to it called "walkdata.csv"

(3) they will produce a csv file called

    walkdata.csv
   
(4) Run the command

    python3 analysis.py
   
   to see the statistical anaylsis and machine learning techiniques.
   
Outputs for this:

    -ANOVA p value
    -Mann-WhitneyU p values
    -Machine Learning Classifiers' Accuracy Score on
        -Decision Tree
        -Naive Bayes
        -K Nearest Neighbours 
     for each of age, gender, and height