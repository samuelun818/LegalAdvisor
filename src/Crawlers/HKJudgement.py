import os
import time

from Crawlers.Judgement import *

from src.Helpers import log_helper, html_helper, file_helper, dataset_helper

class HKJudgement(Judgement):
    def __init__(self):
        super.__init__()
        self.LOCATION = "HK"
        self.JUDGEMENT_URL = 'https://legalref.judiciary.hk/lrs/common/ju/ju_body.jsp'
        self.JUDGEMENT_QUERY = "?DIS={}&AH=&QS=&FN=&currpage=T#"
        self.JUDGMENT_PREFIX = "judgements"

    def downloadJudgements(self, start_no, end_no):
        for num in range(start_no, end_no):
            self.downloadJudgement(num)

        log_helper.print_message("Download completed.")
        return

    def downloadJudgement(self, judgment_no):
        filename = "../{0}/{1}/{2}.html".format(self.JUDGMENT_PREFIX, self.LOCATION, str(judgment_no))

        if os.path.isfile(filename):
            log_helper.print_message('Already exist judgment: ' + str(judgment_no))
            return

        if judgment_no % 100 == 0:
            time.sleep(2)

        query = self.JUDGEMENT_QUERY.format(judgment_no)
        url = self.JUDGEMENT_URL + query
        content = html_helper.get_urlcontent(url)

        if content != None:
            html_helper.write_html(filename, content)
            log_helper.print_message('Save judgment file: {}'.format(filename))

