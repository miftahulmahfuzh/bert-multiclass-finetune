import re

import requests
from googlesearch import search
from tavily import TavilyClient


def search_using_third_party_raw(query):
    print("search_query: " + query)
    tavily_key = "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"
    tavily_client = TavilyClient(api_key=tavily_key)
    raw_res = tavily_client.search(query)['results']
    fin_res = ""
    for res in raw_res:
        fin_res += res['content'] + "\n"
    return fin_res


def search_and_get_web(query):
    # raw_res = search(query, region="id", advanced=True, num_results=3)
    raw_res = search(query, advanced=True, num_results=3)
    fin_res = ""
    for res in raw_res:
        print("URL: " + res.url)
        processed_res = web_crawler(res.url)
        fin_res += processed_res + "\n"
    return fin_res


def web_crawler(url):
    try:
        html_response = requests.get(url).text
        html_body = re.search(r"<body.*?>(.*?)</body>", html_response, re.DOTALL)[0]
        html_content = re.findall(r"<(article|p|div|section).*?>(.*?)</\1>", html_body, re.DOTALL)

        html_element = [(item[0], re.sub('<.*?>', '', re.sub(r'\n|\t|\r| {2}', '', item[1]))) for item in html_content]

        filtered_list = [tup for tup in html_element if len(tup[1].split()) > 3]
        filtered_list = [(x[0], "") if "color" in x[1] else x for x in filtered_list]

        combined_paragrah = [('p', ', '.join(val for key, val in filtered_list if key == 'p'))]
        cleaned_element = [tup for tup in filtered_list if '' not in tup]

        cleaned_element.extend(combined_paragrah)
        maxchar_tuple = max(cleaned_element, key=lambda x: len(''.join(x)))

        return maxchar_tuple[1]

    except:
        return ""
