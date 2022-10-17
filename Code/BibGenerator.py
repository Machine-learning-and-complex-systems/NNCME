# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:25:47 2022

@author: James
"""

from isbnlib import meta
from isbnlib.registry import bibformatters
import sys
import urllib.request
from urllib.error import HTTPError
import time

### PARA DOI
#BASE_URL = 'http://dx.doi.org/'
BASE_URL = 'https://www.doi.org/'

ListaDOI = ['10.1063/1.2159468']

### PARA ISBN
SERVICE = 'openl'
getInfoISBN = bibformatters['bibtex']

ListaISBN = []


#timestamp=time.strftime("%d-%m-%Y-%H%M%S")
timestamp=time.strftime("%d-%m-%Y-%H")
filename='bibliography-'+timestamp+'.txt'
g=open(filename,'a')

for doi in ListaDOI:
    url = BASE_URL + doi
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/x-bibtex')
    time.sleep(2)
    try:
        with urllib.request.urlopen(req) as f:
            bibtexDOI = f.read().decode()
        print(bibtexDOI)
        g.write('\n' + bibtexDOI + '\n\n')

    except HTTPError as e:
        if e.code == 404:
            print('\n DOI {} nao encontrado. DOI not found.\n'.format(doi))
            g.write('DOI: {}'.format(doi)+' não encontrado. Not found.\n\n')
        else:
            print('Serviço indisponível. Service unavailable.')
            sys.exit(1)

for isbn in ListaISBN:
    try:
        bibtexISBN = getInfoISBN(meta(isbn, SERVICE))
        print(bibtexISBN)
        g.write('\n' + bibtexISBN + '\n\n')

    except:
        print('\n ISBN {} nao encontrado. ISBN not found.\n'.format(isbn))
        g.write('ISBN: {}'.format(isbn)+' não encontrado. Not found.\n\n')
g.close()