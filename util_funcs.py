from urllib.parse import urlparse, parse_qs
import re
from os.path import isfile, getsize
import pickle


# Check if file exists
def check_file(path):
    if not isfile(path):
        return False
    return True


# Check if db file exists
def check_db_file(db_name):
    if not isfile(db_name):
        return False
    if getsize(db_name) < 100:  # SQLite database file header is 100 bytes
        return False

    return True


# Return pool index
def get_pool_idx(path):
    if check_file(path) is False:
        pool_idx = 1
    else:
        with open(path, 'rb') as f:
            pool_idx = pickle.load(f)
    return pool_idx

'''
def get_mab_object(path):
    if check_file(path) is False:
    
    else:
        
testval = 2
with open('testfile', 'wb') as f:
    pickle.dump(algorithm, f)

    pickle.dump(testval, f)
    #pickle.dump(algorithm.averages, f)

with open('testfile', 'rb') as f:
    h = pickle.load(f)
    print(h.averages)
    print(h.counts)
    t2 = pickle.load(f)
    print(t2)
    #print(algorithm.averages)
'''

# Update pool index
def update_pool_idx(path):
    pool_idx = get_pool_idx(path)
    pool_idx += 1
    with open(path, 'wb') as f:
        pickle.dump(pool_idx, f)
    return pool_idx


# URL parsing
def parse_url(url):
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)

    sorder = (params['sorder'][0])
    aorder = (params['aorder'][0])
    sid = (params['sid1'][0])
    oid = (params['oid'][0])
    aid = (params['aid'][0])
    id_dic = {'sorder': sorder, 'aorder': aorder, 'sid': sid, 'oid': oid, 'aid': aid}
    return id_dic


# News contents crawling
def get_news_description(soup):
    description = str(soup.find('meta', property='og:description'))
    description = re.sub('<meta content=["|\']', '', description, 0, re.I | re.S)
    description = re.sub('["|\'] property="og:description"/>', '', description, 0, re.I | re.S)
    return description


# News contents crawling
def get_news_text(soup, remove_special_chars=False):
    text = ''
    for item in soup.find_all('div', id='dic_area'):
        text = text + str(item.find_all(text=True))

    email_match = re.search('[a-zA-Z0-9\._+]+@', text)
    if email_match is not None:
        email_loc = email_match.start()
        text = text[:email_loc]

    if remove_special_chars:
        text = re.sub('[a-zA-Z]', '', text)
        text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]', '', text)

    return text


# Number of words in news contents
def get_news_text_number_of_words(soup):
    text = get_news_text(soup, True)
    number_of_words = len(re.findall(r'\w+', text))
    return number_of_words


# News creator crawling
def get_news_creator(soup):
    list = soup.find_all('meta', {'name': 'twitter:creator'})
    if (len(list) == 1):
        creator = str(list[0])
        creator = re.sub('<meta content=["|\']', '', creator, 0, re.I | re.S)
        creator = re.sub('["|\'] name="twitter:creator"/>', '', creator, 0, re.I | re.S)
        return creator
    else:
        return ''


# Get news title features
def get_news_title_features(news_title):
    # 제목 길이
    title_length = len(news_title)

    # 제목[] 등장 유무
    pattern = re.compile('\[.+\]')
    is_emphasis = False
    if pattern.match(news_title):
        is_emphasis = True

    # 말줄임(...) 빈도
    pattern = re.compile('[\.]{2,4}', re.I | re.S)
    pattern_iterator = re.finditer(pattern, news_title)
    num_omitted = 0

    for match in pattern_iterator:
        num_omitted += 1

    feat_dic = {'title_length': title_length, 'is_emphasis': is_emphasis, 'num_omitted': num_omitted}
    return feat_dic
