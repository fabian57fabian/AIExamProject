# AIExamProject
Implementing the perceptron voted for AI exam

Project needs a main folder placed on some drive having these folders and files:

cmc
    test.data
    train.data
iris
    test.data
    train.data
MovementAAL
    *all MovementAAL_RSS_[number].csv
    MovementAAL_target.csv
shuttle
    test.data
    train.data
Wilt
    test.csv
    train.csv

To change directory, change dataPath variable in Tester.py
Every data is given to testData method and is processed by PerceptronVoted class.
TODO: allow to change training and testing size
TODO: place all possible datasets to be tested in an array and choose one at runtime
TODO: verify results with Perceptron algorythm from python libraries
Assignment asks to execute PerceptronVoted on 3 datasets:
-one lineary separable
-two not lineary separable
All 3 datasets have to have more at least 1000 instances