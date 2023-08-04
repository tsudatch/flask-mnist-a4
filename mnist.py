# cording: utf-8

import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras
import cv2
from cv2 import THRESH_BINARY
import math


classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"

# 一次的に切り出した画像を保存するフォルダ
NUM_FOLDER = 'trim-num-file' # プログラムの最後でフォルダ内のファイルをすべて消去します



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.debug = False
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





model = load_model('./model.h5', compile=False)#学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)          
        
            
        # ここから画像を切り出して予測する関数

        #　画像サイズが大きすぎる場合は小さくする
        def size_smaller(img):
            if (img.shape[0] > 4000 or img.shape[1] > 3000):
                img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
            elif (img.shape[0] > 2000 or img.shape[1] > 1000):
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))               
            img_size = img.shape[0]
    
            return img, img_size


        #　背景が白ければ黒に反転する
        def  white_to_black(img):                        
            yl = np.mean(img[0:img.shape[0], 0:img.shape[1] // 20])
            yr = np.mean(img[0:img.shape[0], img.shape[1] - img.shape[1] // 20:img.shape[1]])
            xt = np.mean(img[0:img.shape[0] // 20, 0:img.shape[1]])
            xb = np.mean(img[img.shape[0] - img.shape[0] // 20:img.shape[0], 0:img.shape[1]])
            if (yl > 140 and yl <= 255 or yr > 140 and yr <= 255 
                or xt > 140 and xt <= 255 or xb > 140 and xb <= 255):
                img = cv2.bitwise_not(img)
       
            return img


        #　背景がノイズの入っていない白か黒か判断する
        def noise_detect(img):
            m = []
            yl = np.mean(img[0:img.shape[0], 0:img.shape[1] // 20])
            yr = np.mean(img[0:img.shape[0], img.shape[1] - img.shape[1] // 20:img.shape[1]])
            xt = np.mean(img[0:img.shape[0] // 20, 0:img.shape[1]])
            xb = np.mean(img[img.shape[0] - img.shape[0] // 20:img.shape[0], 0:img.shape[1]])
            yl_min = np.min(img[0:img.shape[0], 0:img.shape[1] // 20])
            yr_min = np.min(img[0:img.shape[0], img.shape[1] - img.shape[1] // 20:img.shape[1]])
            xt_min = np.min(img[0:img.shape[0] // 20, 0:img.shape[1]])
            xb_min = np.min(img[img.shape[0] - img.shape[0] // 20:img.shape[0], 0:img.shape[1]])
            yl_max = np.max(img[0:img.shape[0], 0:img.shape[1] // 20])
            yr_max = np.max(img[0:img.shape[0], img.shape[1] - img.shape[1] // 20:img.shape[1]])
            xt_max = np.max(img[0:img.shape[0] // 20, 0:img.shape[1]])
            xb_max = np.max(img[img.shape[0] - img.shape[0] // 20:img.shape[0], 0:img.shape[1]])
            m_min = [yl_min, yr_min, xt_min, xb_min]
            m_max = [yl_max, yr_max, xt_max, xb_max]
            if (yl == 255 and yr == 255 and xt == 255 and xb == 255):
                screen = "w"
                d_lebel = 255
                d_min = np.min(m_min)
                d_max = np.max(img)
            elif (yl == 0 and yr == 0 and xt == 0 and xb == 0):
                screen = "b"
                d_lebel = 0
                d_min = np.min(m_min)
                d_max = np.max(img)
            else:
                screen = "c"
                d_lebel = (yl + yr + xt + xb) // 4
                d_min = np.min(m_min)
                d_max = np.max(img)
        
            return img, screen, d_lebel, d_min, d_max, xb_min


        #　画像にフレーム状のマスクをかけて主に四隅の影によるノイズを除去する
        def mask(img, screen):              
            if screen == 'w': 
                return img
            elif screen =='b':
                #　画像の横サイズを変数に格納
                x_shape = img.shape[1]  
                #　画像の縦サイズを変数に格納
                y_shape = img.shape[0] 
                #　画像横サイズの１５％を変数に格納しマスクの位置調節で使用
                bold_x = img.shape[1] // 15
                #　画像縦サイズの１５％を変数に格納しマスクの位置調節で使用
                bold_y = img.shape[0] // 15
                #　画像横サイズの1/3を変数に格納しマスクの幅調整で使用
                divided_3 = img.shape[1] // 3
                #　画像縦サイズの1/2を変数に格納
                h = img.shape[0] // 2
                #　画像横サイズの1/2を変数に格納
                w = img.shape[1] // 2
                #　画像の横幅の半分を円の半径とする
                r = img.shape[1] // 2        
                #　時間短縮の為のステップ数を設定
                step = 15 
                #　右下のマスク                  
                for i in range(0, 90, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[y + h + bold_y:, x + w - bold_x:]  = 0  
                #　左下のマスク
                for i in range(90, 180, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[y + h + bold_y:, :x + w + bold_x]  = 0
                #　左上のマスク
                for i in range(180, 270, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[:y + w + bold_x, :x + w + bold_x]  = 0    
                #　右上のマスク
                for i in range(270, 360, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[:y + w + bold_x, x + w - bold_x:]  = 0    
                #　フレームの継ぎ目のマスク
                for i in range(0, bold_x, step):
                     img[divided_3:img.shape[0] - divided_3, 0:i] = 0
                     img[divided_3:img.shape[0], img.shape[1] - bold_x:]  = 0
           
                return img


            elif screen == 'c':
                #　画像の横サイズを変数に格納
                x_shape = img.shape[1]   
                #　画像の縦サイズを変数に格納
                y_shape = img.shape[0] 
                #　画像横サイズの１５％を変数に格納しマスクの位置調節で使用
                bold_x = img.shape[1] // 15
                #　画像縦サイズの１５％を変数に格納しマスクの位置調節で使用
                bold_y = img.shape[0] // 15
                #　画像横サイズの1/3を変数に格納しマスクの幅調整で使用
                divided_3 = img.shape[1] // 3
                #　画像縦サイズの1/2を変数に格納
                h = img.shape[0] // 2
                #　画像横サイズの1/2を変数に格納
                w = img.shape[1] // 2
                #　画像の横幅の半分を円の半径とする
                r = img.shape[1] // 2        
                #　背景が白かったら黒にする
                img = white_to_black(img)
                #　時間短縮の為のステップ数を設定
                step = 15 
                #　右下のマスク                  
                for i in range(0, 90, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[y + h + bold_y:, x + w - bold_x:]  = 0  
                #　左下のマスク
                for i in range(90, 180, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[y + h + bold_y:, :x + w + bold_x]  = 0
                #　左上のマスク
                for i in range(180, 270, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[:y + w + bold_x, :x + w + bold_x]  = 0    
                #　右上のマスク
                for i in range(270, 360, step):
                    x = int(math.cos(math.radians(i)) * r)
                    y = int(math.sin(math.radians(i)) * r)
                    img[:y + w + bold_x, x + w - bold_x:]  = 0    
                #　フレームの継ぎ目のマスク
                for i in range(0, bold_x, step):
                     img[divided_3:img.shape[0] - divided_3, 0:i] = 0
                     img[divided_3:img.shape[0], img.shape[1] - bold_x:]  = 0
           
                return img


        #　キャンバスの暗さに合わせて値を引き陰によるノイズを目立たなくする 
        def smoothing_darkness(img, d_lebel):
            d = int(abs(d_lebel * 0.76))#　影カット閾値
   
            if img.shape[0] != 28:
                #　白の背景がもし暗かったら明るく修正する、キャンバスの大きさに合わせて判断する範囲を調整する 
                if(np.mean(img[0:img.shape[0], 0:img.shape[0] // 12]) > 200 and 
                np.mean(img[0:img.shape[0], 0:img.shape[0] // 12]) < 240):
                   img = img - d 
                elif(np.mean(img[0:img.shape[0], img.shape[1] - img.shape[0] // 12:img.shape[1]]) > 200 and 
                np.mean(img[0:img.shape[0], img.shape[1] - img.shape[0] // 12:img.shape[1]]) < 240):
                    img = img - d
                elif(np.mean(img[0:img.shape[0]//12,0:img.shape[1]])>200 and 
                np.mean(img[0:img.shape[0] // 12, 0:img.shape[1]]) < 240):
                    img = img - d
                elif(np.mean(img[img.shape[0] - img.shape[0] // 12:img.shape[0], 0:img.shape[1]]) > 200 and 
                np.mean(img[img.shape[0] - img.shape[0] // 12:img.shape[0], 0:img.shape[1]]) < 240):
                    img = img - d
                elif(np.mean(img[0:img.shape[0], 0:img.shape[0] // 12]) > 180 and 
                np.mean(img[0:img.shape[0], 0:img.shape[0] // 12]) <= 200):
                   img = img - d
                elif(np.mean(img[0:img.shape[0], img.shape[1] - img.shape[0] // 12:img.shape[1]]) > 180 and 
                np.mean(img[0:img.shape[0], img.shape[1] - img.shape[0] // 12:img.shape[1]]) <= 200):
                    img = img - d
                elif(np.mean(img[0:img.shape[0] // 12, 0:img.shape[1]]) > 180 and 
                np.mean(img[0:img.shape[0] // 12, 0:img.shape[1]]) <= 200):
                    img = img - d
                elif(np.mean(img[img.shape[0] - img.shape[0] // 12:img.shape[0], 0:img.shape[1]]) > 180 and 
                np.mean(img[img.shape[0] - img.shape[0] // 12:img.shape[0], 0:img.shape[1]]) <= 200):
                    img = img - d

            return img 


        # 膨張  
        def dilate_img(img, screen):
            # 背景が白で膨張をすると文字が見えなくなる
            # screenはノイズのない背景が白のデータの場合"w",黒の場合"b"を返す
            # 画面が暗い場合は"c"を返す                       
            # ８近傍フィルタ
            filt = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]],  np.uint8)
            if screen == "w": # 背景が白ならば反転して２回膨張処理を行う
                img = cv2.bitwise_not(img)   
                img = cv2.dilate(img, filt)
                img = cv2.dilate(img, filt)
                screen = 'b'
            elif screen == 'b':
                img = cv2.dilate(img, filt)
                img = cv2.dilate(img, filt)
            else: # 反転無しで２回膨張処理を行う
                #img = cv2.bitwise_not(img)
                img = cv2.dilate(img, filt)
                img = cv2.dilate(img, filt) 
                screen = 'b'
            return img, screen


        def erode_img(img, screen):
            # ノイズのない背景が白のデータの場合"w",黒の場合"b"を返す関数で判定        
            # ８近傍フィルタ
            filt = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]], np.uint8)
            if screen == "w": # 背景が白ならば反転して２回収縮処理を行う
                img = cv2.bitwise_not(img) 
                img = cv2.erode(img, filt) 
                screen = 'b'
            elif screen == 'b':
                img = cv2.erode(img, filt)
            else: # 反転無しで２回収縮処理を行う
                img = cv2.erode(img, filt)
                screen = 'b'

            return img, screen


        # 輪郭を抽出する
        def f_contours(img):      
    
            #　背景が白い場合画像の枠が輪郭としてカウントされるので処理する
            img = white_to_black(img)
    
            # グレースケール
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            # ２値化
            _, th_img = cv2.threshold(gray, 130, 255, THRESH_BINARY)
    
            # 輪郭を抽出する
            contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            # 輪郭のx,y,w,hを格納するリストを作成
    
            n = [] # 輪郭の位置
            p = [] #
            t = 0 #
            thresh = 0.021 #0.028～0.021ノイズカット閾値

            # 輪郭を格納する
            for i in contours: 
               x, y, w, h = cv2.boundingRect(i)
       
               # 縦横比が１０倍以上は削除する（ノイズカット）
               if (w // h > 0.1 and h // w < 10 or w // h < 10 and h // w > 0.1):   
                   # 画像サイズの2.1％以下の輪郭は削除する（ノイズカット）
                   if (w > img.shape[1] * thresh and h > img.shape[0] * thresh): 
                        n.append([x, y, w, h])
    
                        # 抽出した輪郭を左上から順番に並べる
            num_location = sorted(n, key=lambda x:(x[1] // (img.shape[1] // 3), x[0]))
   
            # 外接矩形を描画した画像と座標を返す   
            return img, num_location


        # 切り出した画像を一旦ファイルに保存して認識した結果をリストに保存する
        def create_num_file(img, num_location, screen):
    
            # 切り出し他画像の一時保存ディレクトリ名
            dir_path = './' + NUM_FOLDER + '/'   
    
            # 切り出す数字の余白設定
            if img.shape[0] <= 28: # 画像サイズが(28,28)以下は余白無し
                m1, m2, m3, m4 = 0, 0, 0, 0
            elif img.shape[0] <= 50: # 画像サイズが(50,50)以下は余白1
                m1, m2, m3, m4 = 1, 1, 1, 1
            else: # 画像サイズが大きい場合の余白は10
                m1, m2, m3, m4 = 10, 10, 10, 10
    
            # 画像をファイルに保存する
            if img.shape[0] <= 28 or img.shape[1] <= 28: # (28,28)の場合はそのままファイルに保存する  
                cv2.imwrite(os.path.join(NUM_FOLDER, "num0" + ".png"), img)
            else: 
                # 画像サイズが大きければ数字の数だけ画像を切り出してファイルを保存する 
                for i in range(0, len(num_location)):
                    x, y, w, h = num_location[i]  
            
                    # 画像を正方形に切り取る
                    n1, n2 = abs((h - w) // 2), abs((h - w) // 2)

                    # エラーにならない為の処理    
                    if y - m1 < 0: # 切り取る幅が画像の上をオーバー
                        m1 = 0
                    if y + h + m2 > img.shape[0]: # 切り取る幅が画像の下をオーバー 
                        m2 = 0
                    if x - n1 - m3 < 0: # 切り取る幅が画像の左端をオーバー
                       m3 = 0 
                       n1 = 0
                    if x + w + n2 + m4 > img.shape[1]: # 切り取る幅が画像の右端をオーバー 
                       m4 = 0
                       n2 = 0
            
                    # 画像を余白を付けて切り出す
                    trim_img = img[y - m1:y + h + m2, x - n1 - m3:x + w + n2 + m4]

                    # 切り出された画像の大きさに応じて線を太くする
                    if h > img.shape[0] * 0.9 and w > img.shape[1] * 0.9:
                        #　大きい画像は２度関数を使用
                        trim_img, screen = dilate_img(trim_img, screen) 
                        trim_img, screen = dilate_img(trim_img, screen)
                        # 切り出した画像をそれぞれファイルに保存する
                        cv2.imwrite(os.path.join(NUM_FOLDER, 'num' + str(i) + '.png'), trim_img)
                    elif h > img.shape[0] * 0.5 and w > img.shape[1] * 0.5:
                        # 元画像の半分程度の画像は１度関数を使用
                        trim_img, screen = dilate_img(trim_img, screen)
                        # 切り出した画像をそれぞれファイルに保存する
                        cv2.imwrite(os.path.join(NUM_FOLDER, 'num' + str(i) + '.png'), trim_img)
                    else: 
                        # 切り出した画像をそれぞれファイルに保存する
                        cv2.imwrite(os.path.join(NUM_FOLDER, 'num' + str(i) + '.png'), trim_img)
            return dir_path    


        def from_paint(img, screen):
            # グレースケール化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # matplotlibで表示できるように加工するペイントで作ったPNGはこれが必要
            gray1 = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
            # ２値化
            print(img.shape)
            _, th_img = cv2.threshold(gray1, 130, 255, THRESH_BINARY)
            # ２度膨張処理を行う
            if (img.shape[0] <= 50 or img.shape[1] <= 50):
                d_img = th_img
            else:
                d_img, screen = dilate_img(th_img, screen) 
            # ２度膨張処理を行う
            if (img.shape[0] <= 500 or img.shape[1] <= 500):
                d1_img = d_img
            else:
                d1_img, screen = dilate_img(d_img, screen)
            # ２度膨張処理を行う
            if (img.shape[0] <= 100 or img.shape[1] <= 100):
                d2_img = d1_img
            else:
                d2_img, screen = dilate_img(d1_img, screen)
            # 輪郭を抽出する
            c_img, num_location = f_contours(d2_img)           
            # 画像を切り出してファイルに保存する
            dir_path = create_num_file(c_img, num_location, screen) 
            #print('＊＊＊＊paint＊＊＊＊')

            return dir_path


        def from_camera(img, screen, d_min, d_max, xb_min):                  
            # 四隅のノイズを除去する
            m_img = mask(img, screen)
            # グレースケール化
            gray = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
            # matplotlibで表示できるように加工するペイントで作った画像はこれが必要
            p_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
            # ２値化
            _, th_img = cv2.threshold(p_img, 110, 255, THRESH_BINARY)#130  薄い画像は２０
            # ２度膨張処理を行う
            if (img.shape[0] <= 160 or img.shape[1] <= 160):
                d_img = th_img
            else:
                d_img, screen = dilate_img(th_img, screen)
            # 四隅のノイズを除去する
            if (img.shape[0] > 1000 and img.shape[1] >1000):
                m1_img = mask(d_img, screen)
            else:
                m1_img = d_img
            # 輪郭を抽出する
            c_img, num_location = f_contours(m1_img)
            # 画像を切り出してファイルに保存する
            dir_path = create_num_file(c_img, num_location, screen)
            #print('＊＊＊＊camera＊＊＊＊')

            return dir_path


        def from_scaner(img, screen):
            # 四隅のノイズを除去する
            m_img = mask(img, screen)
            # グレースケール化
            gray = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
            # matplotlibで表示できるように加工するペイントで作ったPNGはこれが必要
            p_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
            # ２値化
            _, th_img = cv2.threshold(p_img, 130, 255, THRESH_BINARY)
            # ２度膨張処理を行う
            if (img.shape[0] <= 50 or img.shape[1] <= 50):
                d_img = th_img
            else:
                d_img, screen = dilate_img(th_img, screen)
            # 収縮処理を行う
            if (img.shape[0] <= 100 or img.shape[1] <= 100):
                e_img = d_img
            else:
                e_img, screen = erode_img(d_img, screen)
            # 四隅のノイズを除去する
            m1_img = mask(e_img, screen)
            # ２度膨張処理を行う
            if (img.shape[0] <= 100 or img.shape[1] <= 100):
                d1_img = m1_img
            else:
                d1_img, screen = dilate_img(m1_img, screen)
            # 輪郭を抽出する
            c_img, num_location = f_contours(d1_img)           
            # 画像を切り出してファイルに保存する
            dir_path = create_num_file(c_img, num_location, screen) 
            #print('＊＊＊＊scaner＊＊＊＊')

            return dir_path


        #　ここから画像を受け取り切り出して予測する

        # 画像を読み込む
        img = cv2.imread('./' + filepath)

        #　画像をNumPy配列に変換
        img = np.array(img)

        # 処理速度向上のために画像サイズを小さくする
        _ ,img_size = size_smaller(img)

        # 主にカメラやスキャナー用
        img, screen, d_lebel, d_min, d_max, xb_min = noise_detect(img)

        # 暗い部分のある画像をスムーズにする
        lite_img = smoothing_darkness(img, d_lebel)

        #　ペイント画像、カメラ画像、スキャナで処理を分ける 
        if img_size != 28 and d_lebel == 0 or d_lebel == 255:
            # ペイントからの画像を処理
            dir_path = from_paint(lite_img, screen)
        elif (img_size != 28 and d_lebel > 160 and d_lebel <190 or d_lebel >3 and d_lebel < 30):
            # カメラから画像を処理
            dir_path = from_camera(lite_img, screen, d_min, d_max, xb_min)
        elif (img_size != 28 and d_lebel >= 190 and d_lebel <255):
            # スキャナーからの画像を処理 
            dir_path = from_scaner(lite_img, screen)  
        elif img_size == 28: #(28,28)サイズの画像は複数文字認識無し
            # 輪郭を抽出する
            _, num_location = f_contours(lite_img)           
            # 画像を切り出してファイルに保存する
            dir_path = create_num_file(lite_img, num_location, screen)


        p = [] # 認識結果を格納するリスト
        q = 0 # 切り出して書き込んだファイル数をカウントする

        # 指定のディレクトリ内のファイルを読み込み数字を認識する
        for i in os.listdir(dir_path)[1:]:
            # 切り取った画像を読み込みむ
            img = cv2.imread(dir_path + i)
            # グレースケール化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #(28,28)にリサイズ            
            img = cv2.resize(img,(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img]) 
 
            # 変換したデータをモデルに渡して予測する
            result = model.predict(data)[0] 
    
            # リストに認識結果を格納する
            if np.max(result) < 0.5: # 認識結果が0.5より小さい場合は(*)を表示する
                p.append('(*)')
            elif np.max(result) >= 0.5: # 認識結果が0.5以上の場合は認識結果を格納する
                predicted = result.argmax()
                p.append(classes[predicted])          
            q = q + 1           
        pred_answer = "これは " + ''.join(p) + " です"
        print(str(pred_answer))
        p=[] # リストをリセットする  

        # dir_pathフォルダ内の切り出した数字ファイルを削除します
        for i in range(q): #書き込んだファイル分だけ消去する
            # dir_path内の一番初めのファイルを削除        
            os.remove(dir_path + os.listdir(dir_path)[1]) # gitkeepファイルを入れたのでインデックスは１
        # アップロードフォルダ内のファイルを削除する
        os.remove('./' + UPLOAD_FOLDER + '/' + os.listdir(UPLOAD_FOLDER)[1])
        return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
   port = int(os.environ.get('PORT', 8080))
   app.run(host ='0.0.0.0',port = port)






       
              
                    
