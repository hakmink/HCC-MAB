
from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import time

import slack_news_bot as newsbot
import common_params
from sklearn import preprocessing

bot = newsbot.SlackNewsBot(common_params.CommonParams())
msg = bot.post_message(True)
df = bot.df
df_encoding = df[['subject_idx', 'is_emphasis', 'num_omitted', 'news_creator']]
le = preprocessing.LabelEncoder()

df_encoding = df_encoding.apply(le.fit_transform)

print(df_encoding)


# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(df_encoding)

# 3. Transform
onehotlabels =  enc.transform(df_encoding).toarray()
print(onehotlabels.shape)
#print(onehotlabels)

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data

'''
subject_idx

df_news_creator = df.iloc[:, 0]

- 카테고리 binary 인코딩(9개)
- title length
- 네이버 조회수
- 본문 길이(문자수)
- 제목 [] 등장 유무
- 제목 ... 갯수
- news provider: 연합, 디스패치, 뉴시스, 티비조선, 뉴스원...
'''