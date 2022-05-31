import pandas as pd
import numpy as np

class Titanic_Feature_Engineering():
    
    def __init__(self, raw_dataframe=None):
        self.data = raw_dataframe
        
        # define map objects for prep
        # we'll just use pandas.map to map numerical values back for the model
        self.model_map_objects = {
            'Sex': {'male':0, 'female':1},
            'Embarked': {'C':0, 'Q':1, 'S':2, 'X':3},
            'has_Cabin?': {False:0,True:1},
            'Cabin_level':{'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7,'X':8},
            'title_group':{'Miss':0, 'Mr':1,'Mrs':2, 'Master':3, 'other':4}
        }
        
        # reverse the map above for easier labeling of charts
        self.model_map_labels = self.reverse_object_map_labels(self.model_map_objects)
        
        # used in title grouping 
        self.title_map = {
            'Capt':'other', 
            'Col':'other', 
            'Don':'other', 
            'Dr':'other',
            'Jonkheer':'other', 
            'Lady':'other',
            'Major':'other', 
            'Ms':'Miss',
            'Mlle':'Miss',
            'Mme':'Mrs',
            'Rev':'other', 
            'Sir':'other', 
            'the Countess':'other',
            'Mr':'Mr', 
            'Miss':'Miss', 
            'Mrs':'Mrs', 
            'Master':'Master'
        }
        
    def clean_prepare_modeling_data(self):
        '''
        Function performs entire cleaning, calculation, & handling of categorical variables for our data
        '''
        prepped_data = self.prep_data(raw_dataframe=self.data)
        modeling_data = self.assign_numerical_values_on_categoricals(prepped_data)
        
        return modeling_data
        
        
    # fill missing values, assemble true / false values / add new features
    def prep_data(self, raw_dataframe=None):
        '''
        This function simply performs all of the cleaning & calculations needed to prepare for modeling.
        '''
        
        ### True / False Values
        raw_dataframe['has_Cabin?'] = ~raw_dataframe['Cabin'].isna()
        
        ### Missing Values
        # Age -> For missing age, we will just populate it with the average
        raw_dataframe.loc[raw_dataframe['Age'].isna(), 'Age'] = int(raw_dataframe['Age'].mean())
        
        # Embarked -> For missing embarked locations, we'll just populate it with 'X'
        raw_dataframe.loc[raw_dataframe['Embarked'].isna(), 'Embarked'] = 'X'
        
        # let's do the same thing for 'Cabin'
        raw_dataframe.loc[raw_dataframe['Cabin'].isna(), 'Cabin'] = 'X'

        
        ### New Features
        # 'is_Child?' -> if 15 or younger, then 1 (True) else 0 (False)
        raw_dataframe['is_Child?'] = raw_dataframe['Age'].apply(lambda age: 1 if age <= 15 else 0)
        
        # 'family_aboard' -> sum of Siblings ('SibSp') aboard & Parent / Children ('Parch')
        raw_dataframe['family_aboard'] = raw_dataframe.apply(lambda row: row['SibSp'] + row['Parch'], axis=1)
        
        # 'num_shared_ticket'-> count of other passengers who share the ticket
        raw_dataframe['num_shared_ticket'] = raw_dataframe['Ticket'].apply(lambda ticket_number: self.find_passengers_on_ticket(ticket_number))
        
        # 'per_person_fare' -> takes the 'Fare' & divides it by the total number on the ticket
        raw_dataframe['per_person_fare'] = raw_dataframe.apply(lambda row: round(row['Fare'] / row['num_shared_ticket'], 2), axis=1)
        
        # 'Cabin_level' -> we can get the general cabin level by the first letter of the cabin
        raw_dataframe['Cabin_level'] = raw_dataframe['Cabin'].apply(lambda value: value[0])
        
        ### Group Titles
        # in this data, the titles generally following this format....
        # 'LAST NAME, TITLE->(MR, MRS, MS, OTHER TITLES). FIRST NAME (ANY ACCOMPANYING PASSENGERS)'
        # So essentially, I'll target whatever is after the comma, and then before the period
        raw_dataframe['title_raw'] = raw_dataframe['Name'].apply(lambda name: name.split(',')[1:][0].split('.')[0].strip())
        
        # now, from the raw titles, let's add the groups
        raw_dataframe['title_group'] = raw_dataframe['title_raw'].map(self.title_map)
        
        return raw_dataframe
    
    def assign_numerical_values_on_categoricals(self, prepped_data):
        '''
        This function takes in the prepared data to: 
            1. Assign Numerical values to categorical variables
            2. Removes Non-numerical columns for the DF
            
        This will return a DataFrame object suitable for our modeling needs.
        '''
        
        # first, let's grab our modeling map object and subset the list of keys
        modeling_keys = list(self.model_map_objects.keys())
        
        # now for each key, we'll map numerical values back on the prepped data
        for key in modeling_keys:
           #print('{} Values before: '.format(key), '\n', '-'*40, '\n', prepped_data[key].head(10), end='')
            prepped_data[key] = prepped_data[key].map(self.model_map_objects[key])
            #print('{} Values after: '.format(key), '\n', '-'*40, '\n', prepped_data[key].head(10), end='')
            
        cols_to_exclude = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare', 'title_raw']
        target_cols = [col for col in prepped_data.columns if col not in cols_to_exclude]
        
        return prepped_data.loc[:, target_cols]   
    
    # functions to be used within prep_data
    def find_passengers_on_ticket(self, ticket_number):
        'Returns the number of Passengers that shared a ticket'
        filtered_df = self.data.loc[self.data['Ticket']==ticket_number]
        return len(filtered_df)
    
    def reverse_object_map_labels(self, model_map):
        'Reverses the inner dict values so graph labels use the text, rather than the numerical values'
        reversed_dict = {}
        
        for label in list(model_map.keys()):
            reversed_dict[label] = dict((value, key) for key, value in model_map[label].items())
        
        return reversed_dict
            