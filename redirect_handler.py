from http.server import HTTPServer, BaseHTTPRequestHandler
import util_funcs
import common_params
import pandas as pd
import pickle


class RedirectHandler(BaseHTTPRequestHandler):

    counter = 0

    def do_GET(self):
        RedirectHandler.counter += 1
        url = self.path

        id_dic = util_funcs.parse_url(url)
        subject_order = int(id_dic['sorder'])
        article_order = int(id_dic['aorder'])
        sid = id_dic['sid']
        oid = id_dic['oid']
        aid = id_dic['aid']
        redirection_url = 'http://news.naver.com/main/read.nhn?oid={}&aid={}'.format(oid, aid)

        print(self.client_address)
        print(self.address_string())
        print(redirection_url)
        self.send_response(301)
        self.send_header('Location', redirection_url)
        self.end_headers()

        # MAB Update
        path = 'log/mab.log'
        with open(path, 'rb') as f:
            algorithm = pickle.load(f)
            trials = pickle.load(f)

        selected_arm = article_order
        algorithm.update(selected_arm, 1)

        with open(path, 'wb') as f:
            trials += 1
            pickle.dump(algorithm, f)
            pickle.dump(trials, f)
            print('trials: {}, selected_arm: {} '
                  'reward_averages_of_arms: {}'.format(trials, selected_arm, algorithm.averages))

        with open('log/user_log.csv', 'a') as f:
            data = {'trials': [trials], 'user_ip': [self.address_string()],
                    'subject_order': [subject_order], 'article_order': [article_order],
                    'sid': [sid], 'aid': [aid], 'oid': [oid],
                    'feat1': [1], 'feat2': [1], 'feat3': [1],
                    'feat4': [1], 'feat5': [1], 'feat6': [1], }
            df = pd.DataFrame(data)
            df.to_csv(f, index=False)


if __name__ == "__main__":
    params = common_params.CommonParams()
    try:
        server = HTTPServer((params.server_address, params.server_port), RedirectHandler)
        print('Starting the redirection server, use <Ctrl-C> to stop')
        # wait forever for incoming http requests
        server.serve_forever()

    except KeyboardInterrupt:
        print("Shutting down the redirection server!")
        server.socket.close()
