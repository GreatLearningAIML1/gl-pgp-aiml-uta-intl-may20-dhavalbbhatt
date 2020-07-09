# ## QUESTION 6 - Business Understanding of the Model
# The model, as represented in Q5 of the project, has predicted about 92% accuracy which customers will
# purchase the personal loan.
# These predictions are based on all the variables provided in the exercise, except ZIP Code. ZIP Code has too much
# variability and wasn't a good way to predict who would purchase the loan or not. (ID is a unique identifier and
# as such not used in predictions).
# Based on the findings, it seems that having an existing account (CD, Securities, etc.) provides a higher probability
# of determining if a person would buy a personal loan or not. The rest of the factors (INCOME, CCAvg, Education, Family
# Size, etc. do help improve the model, but don't influence the outcome significantly.
# The imbalance in the classes also accounts for significant changes in predicting outcomes. While there is no need to
# use the class_weights='balanced' parameter, the imbalance in the classes can be accounted by using a more appropriate
# weighting methodology (as done in Q5 of the project). This seems to predict around 15% of population accepting the
# loan offer. This should align with the marketing philosophy of having more "false positives" given that it will be
# better for the marketing teams to target more individuals as compared to less individuals. Having said that, there is
# significant balance between precision and recall (F1 score of around 0.75 being a good indicator of that) - that
# implies that while we are predicting (at high probablistic threshold of around 0.8) high false positives, we are
# not doing it at the cost of false and true negatives - in other words, the financial return on investment should be
# fairly high with the model that is being prescribed in Q5 of this project.