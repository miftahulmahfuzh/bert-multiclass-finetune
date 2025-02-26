import requests
from langchain_core.tools import tool


@tool
def combined_vapfo_summary(code: str) -> str:
    """
    Check one-day summary stock volume, average price, frequency, and offer.

    Args:
        code (str): Stock code.

    Returns:
        str: Result of the stock volume, average price, frequency, and offer.
    """
    code = code.upper()
    print("Tool: combined_vapfo_summary called " + code)

    raw_res = requests.post(
        'http://10.192.1.245:8080/issuer-directory/price-summary',
        headers={'Content-Type': 'application/json'},
        json={"secCodes": [f"{code}"], "pageNumber": 0, "pageSize": 10000, "sort": "secCode", "direction": "DESC"},
        timeout=10, allow_redirects=True)

    data = raw_res.json()['data']
    res = f'''{code} - volume : {data['content'][0]['volume']}
                {code} - frequency : {data['content'][0]['frequency']}
                {code} - averagePrice : {format_rupiah(str(data['content'][0]['averagePrice']))}
                {code} - offer : {format_rupiah(str(data['content'][0]['offer']))}'''

    return res


@tool
def combined_bvhl_pricemod(code: str) -> str:
    """
    Check real-time stock Bid, Value, High Low, and Closed Price.

    Args:
        code (str): Stock code.

    Returns:
        str: Result of the stock real-time stock Bid, Value, High Low, and Closed Price.
    """
    code = code.upper()
    print("Tool: combined_bvhl_pricemod called " + code)

    raw_res = requests.post(
        'http://10.192.1.245:8080/issuer-directory/price-summary',
        headers={'Content-Type': 'application/json'},
        json={"secCodes": [f"{code}"], "pageNumber": 0, "pageSize": 10000, "sort": "secCode", "direction": "DESC"},
        timeout=10, allow_redirects=True)

    data = raw_res.json()['data']
    res = f'''{code} - bid : {format_rupiah(str(data['content'][0]['bid']))}
            {code} - value : {format_rupiah(str(data['content'][0]['value']))}
            {code} - highprice : {format_rupiah(str(data['content'][0]['highPrice']))}
            {code} - lowprice : {format_rupiah(str(data['content'][0]['lowPrice']))}
            {code} - closedprice : {format_rupiah(str(data['content'][0]['prevPrice']))}'''

    return res


@tool
def historical_lookup(code, start_date, end_date) -> str:
    """
    Use system_date as today's date to calculate start_date and end_date.
    Lookup historical open, high, low, and closed price. If the date is not given, input system_date for start_date and end_date.
    Args:
        code (str): Stock code
        start_date (str) : YYYY-MM-dd
        end_date (str) : YYYY-MM-dd

    Returns:
        list: Result of the stock price.

    """
    # code, start_date, end_date = params.split(",")
    code = code.upper()
    print("Tool: historical_lookup code: " + code + " start_date: " + start_date + " || end_date: " + end_date)

    raw_res = requests.post(
        'http://10.192.1.245:8080/orderbook/basic-trading-data',
        headers={'Content-Type': 'application/json'},
        json={"secCode": code, "startDate": start_date, "endDate": end_date},
        timeout=10, allow_redirects=True)

    data = raw_res.json()['data']
    list_historical = ""

    for element in data:
        historical_dict = {'transactionDate': element['transactionDate'], 'secCode': element['secCode'],
                           'openPrice': format_rupiah(str(element['openPrice'])),
                           'closePrice': format_rupiah(str(element['closePrice'])),
                           'highPrice': format_rupiah(str(element['highPrice'])),
                           'lowPrice': format_rupiah(str(element['lowPrice']))}
        list_historical += str(historical_dict)
    return list_historical


@tool
def company_profile(code: str) -> str:
    """
    Lookup company profile.

    Args:
        code (str): Stock code

    Returns:
        str: return company profile.
    """
    code = code.upper()
    print("Tool: company_profile called " + code)

    raw_res = requests.post(
        'http://10.192.1.226:8082/corporate-profile',
        headers={'Content-Type': 'application/json'},
        json={"secCode": code},
        timeout=10, allow_redirects=True)

    data = raw_res.json()['data']

    res = f'''
            {code} - company name : {data['companyName']},
            {code} - company description : {data['generalInformation']},
            {code} - sector : {data['sector']},
            {code} - subsector : {data['subSector']},
            {code} - company listed share : {data['ipoPrice']}
            '''

    return res


@tool
def shareholder_lookup(code: str) -> str:
    """
    Lookup company shareholder

    Args:
        code (str): Stock code

    Returns:
        list : all shareholder by list
    """
    code = code.upper()
    print("Tool: shareholder_lookup called " + code)

    raw_res = requests.post(
        'http://10.192.1.226:8082/corporate-profile',
        headers={'Content-Type': 'application/json'},
        json={"secCode": code},
        timeout=10, allow_redirects=True)

    shareholder = raw_res.json()['data']['shareHolders']
    shareholder_list = "Shareholder of " + code + ": "

    for element in shareholder:
        shareholder_dict = {'Name': element['name'], 'Amount': int(element['amount']),
                            'Percentage': f'{element["percentage"]}%'}
        shareholder_list += str(shareholder_dict)

    return shareholder_list


@tool
def subsidiary_lookup(code: str) -> str:
    """
    Lookup company subsidiaries.

    Args:
        code (str): Stock code

    Returns:
        list: all company subsidiaries.
    """
    code = code.upper()
    print("Tool: subsidiary_lookup called " + code)

    raw_res = requests.post(
        'http://10.192.1.226:8082/corporate-profile',
        headers={'Content-Type': 'application/json'},
        json={"secCode": code},
        timeout=10, allow_redirects=True)

    subsidiaries = raw_res.json()['data']['subsidiaries']
    subsidiaries_list = ""

    for element in subsidiaries:
        subsidiaries_dict = {'Name': element['name'], 'Percentage': round(float(element['percentage']), 2)}
        subsidiaries_list += str(subsidiaries_dict)

    return subsidiaries_list


def format_rupiah(value: str) -> str:
    clean_value = value.replace(",", "")
    number = round(float(clean_value), 2)
    formatted_value = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {formatted_value}"


def format_decimal(number, separator=".", decimal_separator=","):
    integer_part, decimal_part = f"{number:,.2f}".split(".")
    formatted_number = integer_part.replace(",", separator) + decimal_separator + decimal_part
    return formatted_number

# print(subsidary_lookup("BBNI"))
# print(company_profile("BBNI"))
# print(historical_lookup("BBCA, 2024-09-25, 2024-09-27"))
# print(combined_BVHL_pricemod("BBCA"))
