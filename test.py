import requests

# code = "BBRI"
# raw_res = requests.post(
#     'http://10.192.1.245:8080/orderbook/basic-trading-data',
#     headers={'Content-Type': 'application/json'},
#     json={"secCode": code, "startDate": "2025-01-01", "endDate": "2026-01-01"},
#     timeout=10, allow_redirects=True)
#
# data = raw_res.json()['data']
# list_historical = ""
#
# for element in data:
#     historical_dict = {'transactionDate': element['transactionDate'], 'secCode': element['secCode'],
#                        'openPrice': format_rupiah(str(element['openPrice'])),
#                        'closePrice': format_rupiah(str(element['closePrice'])),
#                        'highPrice': format_rupiah(str(element['highPrice'])),
#                        'lowPrice': format_rupiah(str(element['lowPrice']))}
#     list_historical += str(historical_dict)
# res = list_historical
#
# print(res)
# # print(format_rupiah("9,543.008"))
