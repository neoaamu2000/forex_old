# login.py
import MetaTrader5 as mt5

def initialize_mt5():
    """
    MT5への接続を初期化する。
    接続に成功すればTrue、失敗すればFalseを返す。
    """
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return False
    return True

def shutdown_mt5():
    """
    MT5の接続をシャットダウンする。
    """
    mt5.shutdown()
