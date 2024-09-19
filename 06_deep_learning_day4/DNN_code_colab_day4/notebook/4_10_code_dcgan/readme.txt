・事前準備
dcgan.ipynbのあるディレクトリに学習データの入ったディレクトリと以下の2種類のファイルを用意します。
model_dcgan.py
dcgan_architecture.py

・dcgan.ipynbの使い方
必要に応じてargs_dictを調整し、model = DCGAN(args_dict)でモデルの初期化を行う。
学習データを、train_data = np.load(train_data_path)でロードする。
model.train(train_data)でモデルの学習を開始する。
model.generate_img()で学習済みのモデルから画像を生成する。

・学習の記録について
args_dict["exp_name"]で設定した名前のディレクトリが./expに生成されます。
logsディレクトリにDとGのLossが記録され、tensorboardを用いて可視化できます。
ckptディレクトリにチェックポイントが保存されます。
samplesディレクトリに生成画像が保存されます。

・学習データについて
学習データとして、[データ数, 64, 64, 3(RGBの順)]のサイズのnpyファイルを使用します。
各画素値は[-1,1]に正規化しておく必要があります。
デフォルトの学習データへのパスは./train_data/train_data.npyとなっています。

新規で学習データを作成する際は、./img_dataに64×64の画像群を用意し、
ipynbファイルのセル中にあるconvert_img_to_npy関数によりnpyデータに変換します。