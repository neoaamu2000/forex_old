# session_manager.py
import time
from transitions.extensions import HierarchicalMachine as Machine
from datetime import datetime
from リアルタイム本番用.manage_data import current_price_global, check_no_SMA
from リアルタイム本番用.fibonacci_functions import(detect_extension_reversal,
                                get_out_of_range, 
                                watch_price_in_range,
                                check_touch_line)

# 状態定義（"own_position" を "position" に変更）
states = [
    "created_base_arrow",
    "touched_20",
    "created_new_arrow",
    {"name":"infibos", "children":"has_determined_neck", 'initial': False},
    "has_position",
    "closed"
]

# 遷移定義
transitions = [
    {"trigger": "touch_20", "source": "created_base_arrow", "dest": "touched_20"},
    {"trigger": "create_new_arrow", "source": "touched_20", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": ["created_new_arrow", "infibos_has_determined_neck"], "dest": "infibos"},
    {"trigger": "neck_determine", "source": "infibos", "dest": "infibos_has_determined_neck"},
    {"trigger": "build_position", "source": "infibos_has_determined_neck", "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]

#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, pivot_data, up_trend):
        self.name = name
        self.pivot_data = pivot_data[-2:]  # セッション開始時点のピボットデータのコピー
        self.start_pivot = pivot_data[-1][0] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_pivot = [] #推進波の終わりの最高（安）値を保管。ここ以降を調整波と考えられる。append_pivot_dataで設定される
        self.base_fibo37 = [] #推進波に対してリトレースメント37%を知るための変数
        self.base_fibo70 = []
        self.time_of_goldencross = []
        self.highlow_since_new_arrow = [] #上昇トレンドの調整波の戻しの深さを知るための変数
        self.sml_pivot_data = [] #append_sml_pivot_dataでtouch20以降sml_pivotsを記録

#-------------ネックラインに関する制作-----------------
        self.sml_pivots_after_goldencross = []
        self.potential_neck = []
        self.determined_neck = []

        self.destroy_reqest = False


        if up_trend == "True":
            self.up_trend = True
        if up_trend == "False":
            self.up_trend = False

        self.state_times = {} #stateが移行した時の時刻を記録

        #base_arrowができるまでの最後の波に対してフィボナッチをあて
        #上(下)20%のライン(self.fibo_minus_20)に波が触れてこれば
        #そちらの方向への推進波ととらえることができる
        if up_trend == "True":
            _,_,self.fibo_minus_20, self.fibo_minus_150 = detect_extension_reversal(self.pivot_data[-2:],None, None, 0.2, 1.5)
        if up_trend == "False":
            self.fibo_minus_20, self.fibo_minus_150,_,_ = detect_extension_reversal(self.pivot_data[-2:],-0.2,-1.5,None, None)

        

        # 状態に応じた処理関数のディスパッチテーブルを定義
        self.state_actions = {
            "created_base_arrow": self.handle_created_base_arrow,
            "touched_20": self.handle_touched_20,
            "created_new_arrow": self.handle_created_new_arrow,
            "infibos": self.handle_infibos,
            "infibos_has_determined_neck":self.handle_infibos_has_determined_neck,
            "position": self.handle_has_position,
        }

    def execute_state_action(self, df, sml_df): #現時点ではdf直近100件　df.iloc[-100:]
        """
        現在の状態に対応する処理関数を実行する
        """
        action = self.state_actions.get(self.state)
        if action:
            # 必要に応じて df を渡す
            action(df, sml_df)
        else:
            print(f"{self.name}: No action defined for state {self.state}")

    # 各状態で実行される関数
    def handle_created_base_arrow(self, df, sml_df):
        print(f"{self.name}: Handling 'created_base_arrow'")
        if self.up_trend is True and check_touch_line(self.fibo_minus_20,df.iloc[-1]["high"]):
            self.touch_20()
        elif self.up_trend is False and check_touch_line(self.fibo_minus_20,df.iloc[-1]["low"]) is False:
            self.touch_20()
        else:
            pass

    #矢印形成（append_pivot_data）で詳細条件作成
    def handle_touched_20(self, df, sml_df):
        print(f"{self.name}: Handling 'touched_20'")
        
        

    def handle_created_new_arrow(self, df, sml_df):
        print(f"{self.name}: Handling 'created_new_arrow'")
        if self.up_trend is True and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]) is False:
            #調整の深さを判断するので、推進波に対して37%戻したラインより前回のローソクの最安値(df.iloc[-1]["low"])が低ければ調整完了と判断
            self.touch_37()
        elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]) is True:
            self.touch_37()
        else:
            pass

    #↓を修正して↑になったけど、↓はエントリー時の戻しの深さの最終チェックに使えるかも
    # def handle_created_new_arrow(self, df):
    #     print(f"{self.name}: Handling 'created_new_arrow'")
    #     self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])#推進波の終わり以降の最高（安）値（調整の最も深い価格）を取得
    #     if self.up_trend is True and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]) is False:
    #         self.touch_37()
    #     elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]) is True:
    #         self.touch_37()
    #     else:
    #         pass

#--------------------ここから------------------------------
    
    def handle_infibos(self, df, sml_df):
        print(f"{self.name}: Handling 'infibos'")
        self.price_in_range_while_adjustment(df)
        if self.determined_neck:
                self.neck_determined()
        


    def handle_infibos_has_determined_neck(self, df, sml_df):
        print(f"{self.name}: Handling 'infibos'")
        self.price_in_range_while_adjustment(df)
        self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0]) 
        if self.up_trend is True:
            for neckline in self.determined_neck[:]:
                if current_price_global > neckline[1] and check_no_SMA(current_price_global):
                    self.stop_loss = self.highlow_since_new_arrow[1] - 0.006 #推進波終了以降の最も調整の深い部分から0.006円低い価格を取得
                    pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                    highlow = detect_extension_reversal(pivots_data_to_get_take_profit, higher1_percent=0.32)
                    self.take_profit = highlow[2]
                    ここでbacktestingのエントリーメソッド
                    self.build_position()
                elif current_price_global > neckline[1] and check_no_SMA() is False:
                    self.determined_neck.remove(neckline)
                    if not self.determined_neck:
                        self.touch_37()

        if self.up_trend is False:
            for neckline in self.determined_neck[:]:
                if current_price_global < neckline[1] and check_no_SMA(current_price_global) is False:
                    self.stop_loss = self.highlow_since_new_arrow[0] + 0.006 #推進波終了以降の最も調整の深い部分から0.006円高い価格を取得
                    pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                    highlow = detect_extension_reversal(pivots_data_to_get_take_profit, lower1_percent= -0.32)
                    self.take_profit = highlow[0]
                    ここでbacktestingのエントリーメソッド
                    self.build_position()
                elif current_price_global < neckline[1] and check_no_SMA() is False:
                    self.determined_neck.remove(neckline)
                    if not self.determined_neck:
                        self.touch_37()

        
    
    def handle_has_position(self, df, sml_df):
        print(f"{self.name}: Handling 'position'")

    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------

    def record_state_time(self):
        """状態が遷移した際に状態名と時刻を記録"""
        state_name = self.state
        self.state_times[state_name] = datetime.now()


    def on_enter_created_new_arrow(self):
        """
        ここで推進波に対しての37%の戻りを定義しておく。
        上昇波であればhighに対して37%の戻り
        下降波であればlowに対して37%の戻り
        """
        self.record_state_time()
        print(f"{self.name}: Entered 'created_new_arrow' state.")
        if self.up_trend is True:
            self.base_fibo70,_,self.base_fibo37,_ = detect_extension_reversal(self.pivot_data[-2:],lower1_percent=0.3, higher1_percent=-0.37)
        else:
            self.base_fibo37,_,self.base_fibo70,_ = detect_extension_reversal(self.pivot_data[-2:],lower1_percent=0.37, higher1_percent=-0.3)



    def on_enter_infibos(self):
        self.record_state_time()
        print(f"{self.name}: Entered 'infibos' state.")

    def on_enter_position(self):
        self.record_state_time()
        print(f"{self.name}: Entered 'position' state.")
   
    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"


        

#---------------------------------------------------------------------
#その他の今後も特に使いそうな機能
#---------------------------------------------------------------------

    def should_check():
        """現在の秒数がチェック対象の範囲内か判定する。"""
        # 現在の秒を取得
        sec = datetime.now().second
        # 例えば、58秒〜59秒、もしくは0秒〜3秒の間をチェック対象にする
        return sec >= 54 or sec <= 2
    
    def get_high_and_low_in_term(self, df,time):
        """
        dfから、指定した期間以降の最高値と最安値を検出しreturnする
        """
        if df.index.name != "time":
            df = df.set_index("time", drop=False)
        required_df = df[time:]
        highest_price = required_df[["open", "high", "low", "close"]].max().max()
        lowest_price = required_df[['open', 'high', 'low', 'close']].min().min()
        highest_close = required_df["close"].max()
        lowest_close = required_df["close"].min()

        return highest_price, lowest_price, highest_close, lowest_close

    def get_golden_cross_time(self,df,sml_df):
        """
        dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,20MAの値など)
        sml_dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,4MAの値など)
        基準となるBASE_SMAと一つ下のフラクタル構造のSML_SMAのゴールデンクロスを起こした時間を検出
        調整波の始まり時間をゴールデンクロス基準で把握し、それ以降の最も深い調整位置を知るためのメソッド。
        このゴールデンクロスを起こした後にBASE_SMAが調整方向に転換していてtouch37を満たせば
        調整波として完全に基準を満たしていると判断することができる
        USDJPY 2/19 13:22付近でのエントリーみたいなのをなくすための措置
        """
        df_indexed = df.set_index("time", drop=False)
        sml_df_indexed = sml_df.set_index("time", drop=False)
        base_sma_since_new_arrow = df_indexed.loc[self.new_arrow_pivot[0]:]
        sml_sma_since_new_arrow = sml_df_indexed.loc[self.new_arrow_pivot[0]:]
        if self.up_trend is True:
            for i in range(0, len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] > sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"]
        elif self.up_trend is False:
            for i in range(0, len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] < sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"]
        else:
            return None

    def get_sml_pivots_after_goldencross(self,sml_pivots):
        """
        goldencross以降のsmall_pivotsのデータを取得するメソッド
        """
        for idx, pivot in enumerate(sml_pivots):
            if pivot[0] > self.time_of_goldencross():
                self.sml_pivots_after_goldencross = sml_pivots[idx:]

    def get_potential_neck_wheninto_newarrow(self): 
        """
        ゴールデンクロス以降のsml_pivots(sml_pivots_after_goldencross)を
        sml_pvtsに格納し、その中にネックラインになりうるpivot(上昇トレンド中の調整のhigh
        下降トレンド中の調整のlow)があれば、その次のsml_pvtsの戻しの深さ次第で
        determined_neckに入れる。その価格を超えたらエントリーできる点。
        ネックラインになりうるpivotの次のpivotが生成されてなければ
        potential_neckに格納。(この場合次のpivot確定待ち)
        """
        sml_pvts = self.sml_pivots_after_goldencross

        if len(sml_pvts) >= 2 and self.up_trend is True:
            for i in range(1, len(sml_pvts)):
                if sml_pvts[i][2] == "high" and sml_pvts[i][0] > self.state_times["infibos"]:
                    if i + 1 < len(sml_pvts):
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,-0.32,0.32,None,None)
                        if watch_price_in_range(fibo32_of_ptl_neck[0],fibo32_of_ptl_neck[1],sml_pvts[i+1]):
                            self.determined_neck.append(sml_pvts[i])
                    else:
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,-0.32,0.32,None,None)
                        self.potential_neck.append(sml_pvts[i])

        if len(sml_pvts) >= 2 and self.up_trend is False:
            for i in range(1, len(sml_pvts)):
                if sml_pvts[i][2] == "low" and sml_pvts[i][0] > self.state_times["infibos"]:
                    if i + 1 < len(sml_pvts):
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,None,None,-0.32,0.32)
                        if watch_price_in_range(fibo32_of_ptl_neck[2],fibo32_of_ptl_neck[3],sml_pvts[i+1]):
                            self.determined_neck.append(sml_pvts[i])
                    else:
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,None,None,-0.32,0.32)
                        self.potential_neck.append(sml_pvts[i])

    def price_in_range_while_adjustment(self,df):
        if self.up_trend is True:
            high = self.new_arrow_pivot[1]
            low = self.base_fibo70
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                self.destroy_reqest = True

        elif self.up_trend is False:
            high = self.base_fibo70
            low = self.new_arrow_pivot[1]
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                self.destroy_reqest = True
        



#############################################
# セッションマネージャークラス
#############################################
class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションは session_id をキーに管理
        self.next_session_id = 1


    def add_session(self, pivot_data, up_trend):
        """
        新しいセッションを生成して管理リストに追加する。
        """
        session = MyModel(f"Session_{self.next_session_id}", pivot_data, up_trend)
        self.sessions[self.next_session_id] = session
        print(f"New session created: {session}")
        self.next_session_id += 1
        return session

    def append_pivot_data(self, new_pivot_data, df, sml_df):
        """
        mainでlast_pivot_dataが更新されたら受け取って各セッションのpivot_dataに追加
        touched_20の場合推進波の形成が終わったサインとしてcreated_new_arrowに移る
        "created_base_arrow"の場合touched_20に移る前 (推進波になる前)に波終了でセッション削除
        """
        sessions_to_delete = []

        for session_id, session in self.sessions.items():
            session.pivot_data.append(new_pivot_data)

            if session.state == "touched_20":
                session.time_of_goldencross = session.get_golden_cross_time(df, sml_df)
                session.get_sml_pivots_after_goldencross(session.sml_pivot_data)
                session.new_arrow_pivot = new_pivot_data
                session.get_potential_neck_wheninto_newarrow()
                session.create_new_arrow()
                """
                stateがtouched_20で矢印が出現ということは推進波が終わり20SMAまで調整方向に
                傾いている合図。SMAが傾いているころにはsml_smaは間違いなく
                調整方向へのゴールデンクロス状態。
                ゴールデンクロスが起きた時からBASE_SMAが調整方向に傾いたと検出されるまでにも
                調整の一番深い場所やネックラインとなりうるsml_pivotが生成されている可能性もあるので
                この段階でsml_pivots_after_goldencrossを取得しておく
                """
                
            elif session.state == "created_base_arrow":
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            del self.sessions[session_id]
            print(f"Session {session_id} deleted due to non trade posibility")

    def append_sml_pivot_data(self, new_sml_pivot_data):
        """
        mainでlast_pivot_dataが更新されたら受け取って各セッションのpivot_dataに追加
        touched_20の場合推進波の形成が終わったサインとしてcreated_new_arrowに移る
        "created_base_arrow"の場合touched_20に移る前 (推進波になる前)に波終了でセッション削除
        """
        for session in self.sessions.values():
            if session.state != "created_base_arrow":
                session.sml_pivot_data.append(new_sml_pivot_data)

            if session.new_arrow_pivot is not None:
                print(f"アペンド前:{session.name},{session.sml_pivots_after_goldencross}")
                session.sml_pivots_after_goldencross.append(new_sml_pivot_data)
                print(f"アペンド後:{session.name},{session.sml_pivots_after_goldencross}")

                if session.potential_neck:
                    session.check_potential_to_determine_neck()

                if not session.potential_neck and session.up_trend is True and new_sml_pivot_data[2] == "high":
                    session.potential_neck.append(new_sml_pivot_data)
                elif not session.potential_neck and session.up_trend is False and new_sml_pivot_data[2] == "low":
                    session.potential_neck.append(new_sml_pivot_data)
        # for session in self.sessions.items():
        #     if session.state != "created_base_arrow":
        #         session.sml_pivot_data.append(new_sml_pivot_data)

        #     if session.state == "infibos" and self.potential_neck:
        #         self.potential_neck
            
        


    def send_candle_data_tosession(self,df,sml_df):#dfは直近100件渡されるようになってます
        """
        mainからローソク足データが送信された時に各セッションがstate次に進めないか確認
        """
        sessions_to_delete = []

        for session_id, session in self.sessions.items():
            session.execute_state_action(df, sml_df)
            if session.destroy_reqest is True:
                sessions_to_delete.append(session_id)

        for session_id in sessions_to_delete:
            del self.sessions[session_id]
            print(f"Session {session_id} deleted due to out of range")

        


    

    def check_in_range(self):
        avoid_state = "created_base_arrow", "build_position","position_reached161","position_reached200"
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            if session.state not in avoid_state:
                if session.up_trend == True:
                    result = watch_price_in_range(session.pivot_data[1],session.high150)
                    if result is False:
                        sessions_to_delete.append(session_id)
                else:
                    result = watch_price_in_range(session.pivot_data[0],session.low150)
                    if result is False:
                        sessions_to_delete.append(session_id)
        for session_id in sessions_to_delete:
            del self.sessions[session_id]
            print(f"Session {session_id} deleted due to out-of-range condition.")

    
    def __repr__(self):
        return f"WaveManager(sessions={list(self.sessions.values())})"





import unittest

class TestModel(object):
    def __init__(self, name):
        self.name = name
        # 状態遷移時刻を記録する辞書
        self.state_times = {}
        # 状態遷移マシンの生成（初期状態は "created_base_arrow"）
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        # 各状態に対して共通の on_enter コールバックを登録
        for state in self.machine.states:
            # ここでは state 自体が文字列の場合、そのまま使う
            self.machine.on_enter(state, self.record_state_time)
    
    def record_state_time(self, event=None):
        # event が渡されていれば event.state.name を、なければ現在の状態 (self.state) を使用
        state_name = event.state.name if event is not None else self.state
        self.state_times[state_name] = datetime.now()
        print(f"{self.name}: Entered '{state_name}' at {self.state_times[state_name]}")
    
    def __repr__(self):
        return f"TestModel(name={self.name}, state={self.state})"

class TestStateTransitions(unittest.TestCase):
    def test_transitions(self):
        model = TestModel("TestSession")
        self.assertEqual(model.state, "created_base_arrow")
        
        time.sleep(0.5)  # 少し待って時刻に差をつける
        model.touch_20()
        self.assertEqual(model.state, "touched_20")
        
        time.sleep(0.5)
        model.create_new_arrow()
        self.assertEqual(model.state, "created_new_arrow")
        
        time.sleep(0.5)
        model.touch_37()
        self.assertEqual(model.state, "infibos")

        time.sleep(0.5)
        model.neck_determine()
        self.assertEqual(model.state, "infibos_has_determined_neck")
        
        time.sleep(0.5)
        model.build_position()
        self.assertEqual(model.state, "has_position")
        
        # 状態遷移時刻が記録されているか確認
        # for s in states:
        #     self.assertIn(s, model.state_times)
        
        # すべての状態遷移時刻を出力（デバッグ用）
        print("\nRecorded state times:")
        for state, t in model.state_times.items():
            print(f"{state}: {t}")

if __name__ == '__main__':
    unittest.main()