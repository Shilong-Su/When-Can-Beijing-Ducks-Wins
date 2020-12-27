import pandas as pd

data = pd.read_excel('带进球率2005-2020 -需清洗.xlsx',sheet_name='Sheet1')


data['2分球']=pd.to_numeric(data['2分球'].str[-4:-2])/100
data['3分球']=pd.to_numeric(data['3分球'].str[-4:-2])/100
data['罚球']=pd.to_numeric(data['罚球'].str[-4:-2])/100



data.to_excel('带进球率2005-2020 -需清洗胜负.xlsx')





