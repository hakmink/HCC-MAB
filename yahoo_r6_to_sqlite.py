# yahoo r6 data -> sqlite3 database

import datetime
import os
import time

from yahoo_r6.process_to_sqlite import ProcessWebscope

# process all files in the data directory
proc = ProcessWebscope('yahoo_r6_full.db', log=False)

t0 = time.time()
for file in os.listdir('Webscope/R6/'):
    if file.endswith('.gz'):
        print(file)
        proc.process_file('Webscope/R6/' + file)
# proc.process_file('Webscope/R6/ydata-fp-td-clicks-v1_0.20090501.gz')
# proc.process_file('Webscope/R6/ydata-fp-td-clicks-v1_0.20090501.gz', skip_lines=4600000)  # test
t1 = time.time()

the_time = str(datetime.timedelta(seconds=t1-t0))
print('Processing all files took a total of {}'.format(the_time))
