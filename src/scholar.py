import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from IPython.display import display

from .utils import remove_middle_names


class Scholar(object):
    def __init__(self, name, referer_id=None):
        self.name = name
        if referer_id is not None:
            referer_url = Scholar._id_to_url(referer_id)
        else:
            referer_url = None
        self.id = self._get_author_id(referer_url=referer_url)
        self.html = self._parse_profile_html(referer_url=referer_url)
        self.citations = self._get_citations()
        self.metrics = self._get_metrics()
        del self.html


    def __repr__(self):
        return f'Scholar — {self.name}'


    @staticmethod
    def _extract_texts_as_ints(l):
        return [int(x.text) for x in l]


    @staticmethod
    def _get_html(url, referer_url=None):
        headers = {
            'accept': ('text/html,application/xhtml+xml,application/'
                       'xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.8',
            'upgrade-insecure-requests': '1',
            'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                           '61.0.3163.100 Safari/537.36'),
        }

        if referer_url is not None and isinstance(referer_url, str):
            headers['referer'] = referer_url

        with requests.Session() as s:
            r = s.get(url, headers=headers)

        if r.status_code != 200:
            raise ConnectionError(f'Failed to get data from\n{url}')

        return BeautifulSoup(r.text, 'html.parser')


    @staticmethod
    def _name_to_url(name):
        parsed = '+'.join(name.lower().split())
        return ('https://scholar.google.com/citations?hl=en'
                f'&view_op=search_authors&mauthors={parsed}')


    @staticmethod
    def _id_to_url(id):
        return f'https://scholar.google.com/citations?hl=en&user={id}'


    def _get_author_id(self, referer_url=None):
        url = Scholar._name_to_url(self.name)
        soup = Scholar._get_html(url, referer_url=referer_url)
        res = soup.find_all('a', attrs={'class': 'gs_ai_pho'})

        if not len(res):
            raise LookupError(f'No scholar profile found for {self.name}.')
        found_scholar = False
        for r in res:
            name = soup.find('h3', attrs={'class': 'gs_ai_name'}).text
            if (remove_middle_names(name).lower() ==
                    remove_middle_names(self.name).lower()):
                found_scholar = True
                res = r
        if not found_scholar:
            raise LookupError(f'No matching scholar name.')

        return res.attrs['href'].split('user=')[-1]


    def _parse_profile_html(self, referer_url=None):
        url = Scholar._id_to_url(self.id)
        return Scholar._get_html(url, referer_url=referer_url)


    def _get_metrics(self):
        index = ['Citations', 'h-index', 'i10-index']
        columns = ['All', 'Since 2014']
        scores = Scholar._extract_texts_as_ints(
            self.html.find_all('td', attrs={'class': 'gsc_rsb_std'}))

        vals = []
        for i in range(0, len(scores), 2):
            vals.append([scores[i], scores[i + 1]])

        return pd.DataFrame(vals, columns=columns, index=index)


    def _get_citations(self):
        years = Scholar._extract_texts_as_ints(
            self.html.find_all('span', attrs={'class': 'gsc_g_t'}))
        citations = Scholar._extract_texts_as_ints(
            self.html.find_all('span', attrs={'class': 'gsc_g_al'}))
        return dict(zip(years, citations))
