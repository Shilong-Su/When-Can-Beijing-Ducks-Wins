import pandas as pd

# data = pd.read_excel('opponent_info（无时间）.xlsx',sheet_name='Tabelle1')

#第一次清洗——进球率转换为[0,1]间小数
# data['2分（京）']=pd.to_numeric(data['2分（京）'].str[:-1])/100
# data['3分（京）']=pd.to_numeric(data['3分（京）'].str[:-1])/100
# data['罚球（京）']=pd.to_numeric(data['罚球（京）'].str[:-1])/100
# data['2分（对）']=pd.to_numeric(data['2分（对）'].str[:-1])/100
# data['3分（对）']=pd.to_numeric(data['3分（对）'].str[:-1])/100
# data['罚球（对）']=pd.to_numeric(data['罚球（对）'].str[:-1])/100
# data.to_excel('opponent_info（无时间，进球率已清洗）.xlsx')
# 获得差值
data = pd.read_excel('opponent_info（无时间，进球率已清洗）.xlsx',sheet_name='Sheet1')
data['2分（差）']=data['2分（京）']-data['2分（对）']
data['3分（差）']=data['3分（京）']-data['3分（对）']
data['罚球（差）']=data['罚球（京）']-data['罚球（对）']
data['进攻篮板（差）']=data['进攻篮板（京）']-data['进攻篮板（对）']
data['防守篮板（差）']=data['防守篮板（京）']-data['防守篮板（对）']
data['助攻（差）']=data['助攻（京）']-data['助攻（对）']
data['犯规（差）']=data['犯规（京）']-data['犯规（对）']
data['抢断（差）']=data['抢断（京）']-data['抢断（对）']
data['失误（差）']=data['失误（京）']-data['失误（对）']
data['扣篮（差）']=data['扣篮（京）']-data['扣篮（对）']
data['被侵（差）']=data['被侵（京）']-data['被侵（对）']
data['得分（差）']=data['得分（京）']-data['得分（对）']
data['胜负']= None
for i in range(len(data['得分（差）'])):
    if data['得分（差）'][i]>0:
        data['胜负'][i] = 1
    else:
        data['胜负'][i] = 0

data.to_excel('opponent_info（无时间，进球率已清洗，已创建分差等，快攻未处理）.xlsx')