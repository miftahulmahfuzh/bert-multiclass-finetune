import requests

def format_rupiah(value: str) -> str:
    clean_value = value.replace(",", "")
    number = round(float(clean_value), 2)
    formatted_value = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {formatted_value}"

def get_stock_price(code: str) -> str:
    """
    Check real-time stock price,
    you can get the latest price of a stock too.

    example query:
    get me the latest BBCA's stock price

    Args:
        code (str): Stock code (e.g., "AAPL").

    Returns:
        str: Result of the stock price.
    """
    code = code.upper()
    print("Tool: stock_price called " + code)
    raw_res = requests.get(
        'https://ui-testing002.istock.co.id/stocks-ui/individual-stock/stock/get-stock-basic-info?accessToken=mockPass&code=' + code)
    res = f"Failed to get stock price of Stock Code: '{code}' from stock_price tool"
    if raw_res:
        data = raw_res.json()['data']
        if isinstance(data, dict):
            print(f"DATA: {data}")
            if 'name' in data:
                if data['name']:
                    res = "Latest price, stock_name: " + data['name'] + ", stock_price: " + format_rupiah(data['price']) + ", change: " + data[
            'chg'] + ", change_percent: " + data['chgPercent']
    return res
