import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import re
from datetime import datetime, timedelta
import csv
import util_funcs
import common_params
import database


def apply_article_idx(row):
    return "{}&aorder={}".format(row["redirection_url"], row["article_idx"])


class NewsScrapper(object):
    """
    Naver Mobile News Scrapper
    """
    def __init__(self, common_params, n_days_ago=0):
        self.param = common_params

        # 최근 -n일자 뉴스 스크래핑
        #  (0: 금일 날짜, 1: 어제, 2: 그제 ...)
        self.date_n_days_ago = (datetime.now() - timedelta(days=n_days_ago)).strftime("%Y%m%d")

        # 네이버 모바일 뉴스 URL
        self.news_ranking_url = self.param.news_main_url + '/rankingList.nhn?sid1=&date='
        self.news_ranking_url += self.date_n_days_ago

        # 뉴스 본문 저장 (임시 테스트용 파일)
        self.news_content_filename = 'ranking_news_'
        self.news_content_filename += self.date_n_days_ago
        self.news_content_filename += '.txt'

        # 뉴스 데이터를 저장하기 위한 데이터프레임 객체
        self.news_df = pd.DataFrame()

        # 뉴스 pool 인덱스
        self.curr_pool_idx = 0

    """
    Scrap news articles
    """
    def scrap(self, verbose=False):
        source_code_from_url = urllib.request.urlopen(self.news_ranking_url)
        soup = BeautifulSoup(source_code_from_url, 'lxml', from_encoding='utf-8')

        # Find all news subjects such as politics, economy, and so on
        items = soup.find_all('li', id=re.compile(self.param.news_subject_regexp))

        # 뉴스 기사 index (0~26)
        article_idx = 0
        # 뉴스 주제 index (0~8)
        subject_idx = 0

        #news_content_file = open(self.news_content_filename, 'w')
        csv_file = open(self.param.csv_path, 'w', encoding='utf=8', newline='')
        csv_writer = csv.writer(csv_file)

        for item in items:
            # 각 주제별 뉴스는 3건이므로
            if article_idx % 3 == 0:
                if verbose:
                    print(self.param.news_subject[subject_idx])
                subject_idx += 1

            # 뉴스 제목
            title = (item.find('div', class_='commonlist_tx_headline')).get_text()

            # 클릭 횟수
            visit = (item.find('div', class_='commonlist_tx_visit')).get_text()

            # 썸네일 이미지 링크
            image = item.find('div', class_='commonlist_img')
            if image is not None:
                image = re.findall(re.compile('src="(.*)" '), str(image))
                thumbnail_url = image[0]
            else:
                thumbnail_url = None

            # 조회 수
            visit = int(visit[3:].replace(',', '')) # 조회수1000 => 1000
            title_link = item.select('a')
            news_url = title_link[0]['href']
            news_query = news_url[17:]

            # 원본 기사 URL
            article_url = self.param.news_main_url + news_url

            # 리다이렉션 서버 URL
            redirection_url = 'http://{}:{}/alab.ml?sorder={}&{}'.\
                format(self.param.server_address, self.param.server_port, subject_idx-1, news_query)

            # 뉴스 ID (네이버 뉴스는 oid, aid 만으로 접근 가능)
            oid = news_query[4:7]
            aid = news_query[12:22]

            # 뉴스 본문 주소 크롤링
            source_code_from_url = urllib.request.urlopen(article_url)
            soup = BeautifulSoup(source_code_from_url, 'lxml', from_encoding='utf-8')

            # 뉴스 요약
            news_description = util_funcs.get_news_description(soup)

            # 뉴스 본문 문자 수
            news_number_of_words = util_funcs.get_news_text_number_of_words(soup)

            # 뉴스 feature; 제목 길이, 제목[] 등장 유무, 말줄임(...) 빈도
            title_feature = util_funcs.get_news_title_features(title)

            # 뉴스 제목 길이
            title_length = title_feature['title_length']

            # 뉴스 제목 내 [] 패턴 등장 여부
            is_emphasis = title_feature['is_emphasis']

            # 뉴스 제목 내 ... 패턴 발생 횟수
            num_omitted = title_feature['num_omitted']

            # 뉴스 작성 언론사
            news_creator = util_funcs.get_news_creator(soup)

            csv_writer.writerow([subject_idx-1, article_idx, title, news_description,
                                 news_url, redirection_url, oid, aid,
                                 title_length, visit, news_number_of_words, is_emphasis,
                                 num_omitted, news_creator, thumbnail_url])

            if verbose:
                print(title)
                print(news_url)
                print(news_query)
                print(article_url)
                print(redirection_url)
                print()

            #text = get_news_text(article_url)
            #text = clean_news_text(text)
            #news_content_file.write(text)

            article_idx += 1

        #news_content_file.close()
        csv_file.close()

        df = self.param.load_news_df()

        # 중복 뉴스 제거 및 csv 갱신
        df = df.drop_duplicates(['aid'], keep='first')
        df.reset_index(drop=True, inplace=True)
        df.drop(df.columns[[1]], axis=1, inplace=True)
        df.insert(1, 'article_idx', range(0, len(df)))
        # Redirection URL 수정 (중복 뉴스 제거로 인한 arm index 수정 반영)
        df['redirection_url'] = df.apply(apply_article_idx, axis=1)
        df.to_csv(self.param.csv_path, index=False, header=False)

        is_oldnews_exist = util_funcs.check_file(self.param.csv_path_old)

        if is_oldnews_exist is True:
            df_old = self.param.load_news_df(False)
            df_prev_news_id = df_old.iloc[:, 6:8]
            df_curr_news_id = df.iloc[:, 6:8]

            # 뉴스 목록 변경 시, 뉴스 pool 업데이트
            if not df_prev_news_id.equals(df_curr_news_id):
                # 뉴스 pool index; 뉴스 스크래핑 시 추천 뉴스 목록이 바뀔 때마다 pool index 증가
                self.curr_pool_idx = util_funcs.update_pool_idx(self.param.pool_idx_path)
            else:
                self.curr_pool_idx = util_funcs.get_pool_idx(self.param.pool_idx_path)
        else:
            self.curr_pool_idx = util_funcs.get_pool_idx(self.param.pool_idx_path)

        df.to_csv(self.param.csv_path_old, index=False, header=False)

        return df

    def get_curr_pool_idx(self):
        return self.curr_pool_idx


if __name__ == "__main__":
    param = common_params.CommonParams()
    scrapper = NewsScrapper(param, 0)
    news_df = scrapper.scrap()

    db = database.NewsDatabase(param)
    db.open_db()
    is_pool_changed = db.insert_article(scrapper.get_curr_pool_idx())
    db.close_db()
