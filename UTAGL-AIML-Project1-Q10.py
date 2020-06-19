#########################################
#       IMPORT NECESSARY PACKAGES       #
#########################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

#########################################
#           READ DATA FRAME             #
#########################################
data = pd.read_csv('Data.csv')
item = pd.read_csv('item.csv')
user = pd.read_csv('user.csv')

#########################################
#                 EDA                   #
#########################################

def createDF ():
    import pandas as pd
    df1 = pd.merge(item, data, how='outer')
    total_df = pd.merge(df1, user, how='outer')
    return(total_df)

## FUNCTION 2 evaluate hypothesis
def checkHypo(genre):
    total_df = createDF()

    b = pd.DataFrame(total_df[total_df[genre] == 1].groupby(['user id', 'gender']).size())
    c = pd.DataFrame(b.groupby('gender').size())
    unique_females = c.loc['F']
    unique_males = c.loc['M']

    if unique_females[0] > unique_males[0]:
        print("More FEMALES watch {} than males".format(genre))
        if genre == 'Drama':
            print("Hypothesis is WRONG")
        elif genre == 'Romance':
            print("Hypothesis is WRONG")
        elif genre == 'Sci-Fi':
            print("Hypothesis is RIGHT")
        else:
            return
    else:
        print("More MALES watch {} than females".format(genre))
        if genre == 'Drama':
            print("Hypothesis is RIGHT")
        elif genre == 'Romance':
            print("Hypothesis is RIGHT")
        elif genre == 'Sci-Fi':
            print("Hypothesis is WRONG")
        else:
            return

    return

##### CHECK HYPOTHESIS -

#### HYPOTHESIS 1: Men Watch more "DRAMA" than Women ---> TRUE
checkHypo('Drama')
print("*" * 50)
checkHypo('Sci-Fi')
print("*" * 50)
checkHypo('Romance')
print("*" * 50)