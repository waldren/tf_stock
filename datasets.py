'''
Contains code to generate training datasets.
'''
import yfinance as yf
def get_stock_price_history(symbol, start_date, end_date):
    s = yf.Ticker(symbol)
    return s.history(start=start_date, end=end_date)
class Generator():
    def __init__(self, symbols=[]):
        self.symbols = symbols
    
    def create_datasets():
        pass
def test():
    symb = 'AAPL'
    start ='2000-01-01'
    end ='2021-06-16'
    df = get_stock_price_history(symb, start, end)
    print(df.tail())
if __name__ == "__main__":
    test()