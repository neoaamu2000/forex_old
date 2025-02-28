def detect_extension_reversal(
    pivot_data,
):
    """
    pivot_data: [(datetime, price, type), ...] の形式を想定

    オプションのパーセンテージパラメータがすべて None であれば、デフォルトの拡張レベルをそのまま返す。
    デフォルトの拡張レベル（絶対値）は以下の通り：
        上側（高値側）: 160, 139, 120, 80, 60, 67, 60, 37, 20
        下側（安値側）: -20, -37, -60, -67, -80, -120, -139, -160
    ※ ここではユーザー指定の数値そのものを返します。

    オプションパラメーターがひとつでも指定されている場合は、pivot_dataから動的に計算して返します。
    例として、fibo37_percent（例: 0.37）であれば、上側は
       high_val + wave_range * 0.37
    で計算し、fibo37_percentが負の場合は
       low_val + wave_range * fibo37_percent
    で計算する、といった方式とします。（他のパラメータについても同様）
    """


    # オプションパラメーターが指定されている場合は動的に計算する
    if len(pivot_data) < 2:
        # 十分なpivotデータがなければすべてNone
        num = 17  # 17個返す
        return tuple([None] * num)
    
    # 直近2件のpivotの価格を取り出す
    price1 = pivot_data[-2][1]
    price2 = pivot_data[-1][1]
    
    # 高い方・低い方を決定
    high_val = max(price1, price2)
    low_val = min(price1, price2)
    
    wave_range = high_val - low_val
    
    up_factors = [0.0, 0.20, 0.37, 0.60, 0.67, 0.80, 1.20, 1.39, 1.60]
    down_factors = [-0.20, -0.37, -0.60, -0.67, -0.80, -1.20, -1.39, -1.60]
    
    up_levels = [ high_val + wave_range * f for f in up_factors ]
    down_levels = [ low_val + wave_range * f for f in down_factors ]
    
    # ここで、各レベルを整数（または丸めた値）にするかそのままにするかはお好みで
    # 例として、丸めずそのまま返す
    # 最終的なリストは、上側を降順（大きい順）、下側を昇順（大きい方から小さい方）
    # あるいは、ユーザーが示した順序（上側：160,139,120,80,60,67,60,37,20, 下側：-20,-37,...,-160）に合わせるため、
    # ここでは上側を [高い方→低い方] に並べ替え、下側はそのまま逆順に並べたものとします。
    up_levels_sorted = sorted(up_levels, reverse=True)
    down_levels_sorted = sorted(down_levels)
    
    # 最終的に上側と下側を連結して返す（合計で 9+8=17 個になる）
    final_levels = tuple(up_levels_sorted + down_levels_sorted)
    return final_levels
