import pandas as pd
import numpy as np
from pandas import DataFrame

dataset=pd.read_csv("CNPC_1401-1509_DI_v1_1_2016-03-01.csv")

to_drop=['course_id_DI','discipline','userid_DI','registered','viewed','explored','grade','grade_reqs','completed_%','course_reqs','final_cc_cname_DI','primary_reason','LoE_DI','age_DI','gender','start_time_DI','course_start','course_end','last_event_DI','ncontent','nforum_posts','course_length']
dataset.drop(to_drop, inplace=True, axis=1)
dataset=pd.DataFrame(dataset)
dataset['learner_type']=dataset['learner_type'].replace({'Missing':'Passive','Passive participant':'Passive','Active participant':'Active','Drop-in':'Active','Observer':'Active'})
dataset['learner_type']=dataset['learner_type'].fillna('Passive')
dataset['expected_hours_week']= dataset['expected_hours_week'].replace({'Between 4 and 6 hours':5,'Between 2 and 4 hours':3,'Between 1 and 2 hours':2,'Less than 1 hour':1,'Between 6 and 8 hours':7,'More than 8 hours per week':10,'Missing':0})
dataset['expected_hours_week']= dataset['expected_hours_week'].fillna(0)
#print dataset.head()
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(0,50),1)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(50,100),2)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100,500),3)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500,1000),4)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(1000,5000),5)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(5000,10000),6)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(10000,100000),7)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100000,500000),8)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500000,1000000),9)
dataset['nevents']=dataset['nevents'].fillna(0)
dataset['ndays_act']=dataset['ndays_act'].fillna(0)
print np.unique(dataset['ndays_act'])
print dataset.head()