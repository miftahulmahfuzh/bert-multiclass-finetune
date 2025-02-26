# %% IMPORTS
import os

from tool.tool_search_web import web_crawler

tavily_key = "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"
os.environ["TAVILY_API_KEY"] = tavily_key

# def search_and_get_web(query):
#     raw_res = search(query, region="id", advanced=True, num_results=3)
#     fin_res = ""
#     for res in raw_res:
#         buffer = BytesIO()
#         curl = pycurl.Curl()
#         curl.setopt(curl.URL, res.url)
#         curl.setopt(curl.WRITEDATA, buffer)
#         curl.perform()
#         curl.close()
#         body = buffer.getvalue()
#         soup = BeautifulSoup(body.decode('iso-8859-1'), 'html.parser')
#         web_body = soup.getText
#         # web_content = web_body.
#         # for p in paragraphs:
#         #     web_content += p.text
#         fin_res += str(web_body) + "\n"
#     return fin_res

print(web_crawler("https://edition.cnn.com/"))
