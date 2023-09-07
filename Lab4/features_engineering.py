import re
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup

sampledURL = pd.read_csv('allurl.csv')

cleanedURL = pd.DataFrame({'url': list(sampledURL['URLs'].dropna())})

allURLs = pd.DataFrame()
allURLs = allURLs.assign(url=sampledURL['phish_detail_url'].dropna())
allURLs = pd.concat([cleanedURL, allURLs]).reset_index(drop=True)

data_set = {}
for url in allURLs['url']:
    # Stores the response of the given URL
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        response = ""
        soup = -999
        
    domain = urlparse(url).netloc
    if re.match(r"^www.",domain):
        domain = domain.replace("www.","")
    print(domain)
    
    if soup == -999:
        favicon = -1
    else:
        try:
            for head in soup.find_all('head'):
                for head.link in soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                    if url in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                        favicon = 1
                        raise StopIteration
                    else:
                        favicon = -1
                        raise StopIteration
        except StopIteration:
            pass
        
    tmp = data_set.get('favicon', [])
    tmp.append(favicon)
    data_set['favicon'] = tmp
    
    if len(re.findall("\.", domain)) == 1:
        subDomain = 1
    elif len(re.findall("\.", domain)) == 2:
        subDomain = 0
    else:
        subDomain = -1
        
    tmp = data_set.get('subDomain', [])
    tmp.append(subDomain)
    data_set['subDomain'] = tmp

    
results = pd.read_csv('urldata.csv')
results['favicon'] = data_set['favicon']
results['subDomain'] = data_set['subDomain']

results.to_csv('with_new_features.csv', index=False)
