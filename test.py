import pandas as pd
trials = 1
subject_order = 111
article_order = 200
sid = 1
aid = 2
oid = 3
user_ip = "100.10.1.01"

with open('log/user_log5.csv', 'a') as f:

#    df = pd.DataFrame([trials, 1, subject_order, article_order, sid, aid, oid, 1, 1, 1, 1, 1, 1])

    #df = pd.DataFrame()
    #df = df.append({'trials': 1, 'subject_order': 1, 'article_order':1,
    #                'sid': sid, 'aid': aid, 'oid': oid}, ignore_index=True)
    #df = pd.DataFrame([1,4,5,6], columns=['a','b','c','d'])
    data = {'trials': [trials], 'user_ip': [user_ip],
            'subject_order': [subject_order], 'article_order': [article_order],
            'sid': [sid], 'aid': [aid], 'oid': [oid],
            'feat1': [1], 'feat2': [1], 'feat3': [1],
            'feat4': [1], 'feat5': [1], 'feat6': [1],}
    df = pd.DataFrame(data)
    df.to_csv(f, index=False)
