# NDLOCR-Liteアプリケーションのリポジトリ

NDLOCR-Liteを利用してテキスト化を実行するためのアプリケーションを提供するリポジトリです。

NDLOCR-Liteは、[NDLOCR](https://github.com/ndl-lab/ndlocr_cli)の軽量版を目指して開発したOCRであり、ノートパソコン等の一般的な家庭用コンピュータやOS環境で、図書や雑誌といった資料のデジタル化画像からテキストデータが作成できるOCRです。

GPUを必要としないOCR処理に特徴があり、ノートパソコン等の一般的な家庭用コンピュータやOS環境において高速に実行可能です。

Windows(Windows 11)、Mac(Apple M4, macOS Sequoia)及びLinux(Ubuntu 22.04)環境において動作確認しています。

本プログラムは[NDLラボ](https://lab.ndl.go.jp)におけるこれまでの調査研究活動によって得られた知見、特に[NDL古典籍OCR-Lite](https://github.com/ndl-lab/ndlkotenocr-lite)の開発経験を踏まえて職員が内製で開発しました。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については[LICENCE](./LICENCE)をご覧ください。なお、本アプリケーションの実行時に利用するライブラリ等のライセンスについては[LICENCE_DEPENDENCIES](./LICENCE_DEPENDENCEIES)をご覧ください。


## デスクトップアプリケーションによる利用

**デスクトップアプリケーションを利用する際には、日本語（全角文字）を含まないパスにアプリケーションを配置してください。全角文字を含む場合に起動しないことがあります。**

[releases](https://github.com/ndl-lab/ndlocr-lite/releases)からお使いのOS環境（Windows/Mac/Linux）に合ったファイルをダウンロードしてください。


デスクトップアプリケーションの操作方法及びビルド方法については[デスクトップアプリケーションの利用方法](./ndlocr-lite-gui/README.md)を参照してください。


次のgifアニメーションは、

[国立国会図書館総務部総務課 編『国立国会図書館年報』昭和27年度,国立国会図書館,1954. 国立国会図書館デジタルコレクション https://dl.ndl.go.jp/pid/3048008"](https://dl.ndl.go.jp/pid/3048008/1/24)

をNDLOCR-Liteの画面キャプチャ機能によって、画像ファイルを介さずにテキスト化するデモを示しています。

<img src="resource/demo_lite.gif" width="600">

## コマンドラインからの利用
※コマンドラインから操作を行うにはPython 3.10以上が必要です。

事前準備
```
git clone https://github.com/ndl-lab/ndlkotenocr-lite
cd ndlkotenocr-lite
pip install -r requirements.txt
cd src
```
実行例1.（同階層にある「竜宮苦界玉手箱 _ 3巻_9892834_0001」という名称のディレクトリ内の画像を一括処理し、tmpdirという名称のディレクトリに結果を出力する。）
```
python3 ocr.py --sourcedir 竜宮苦界玉手箱 _ 3巻_9892834_0001 --output tmpdir 
```

実行例2.（同階層にある「digidepo_1287221_00000002.jpg」という名称の画像を処理し、tmpdirという名称のディレクトリに結果を出力する。）
```
python3 ocr.py --sourceimg digidepo_1287221_00000002.jpg --output tmpdir 
```

### パラメータの説明

#### `--sourcedir`オプション
処理したい画像の含まれるディレクトリを絶対パスまたは相対パスで指定する。ディレクトリ内の"jpg（jpegも可）"、"png"、"tiff（tifも可）"、"jp2"及び"bmp"の拡張子のファイルを順次処理する。

#### `--sourceimg`オプション
処理したい画像を絶対パスまたは相対パスで直接指定する。"jpg（jpegも可）"、"png"、"tiff（tifも可）"、"jp2"及び"bmp"の拡張子のファイルを処理することが可能。

#### `--output`オプション
OCR結果を保存する出力先ディレクトリを相対パスまたは絶対パスで指定する。

#### `--viz`オプション
`--viz True`を指定することで、文字認識箇所を青枠で表示した画像を出力先ディレクトリに出力する。

#### `--device`オプション（ベータ）
対応GPUを搭載したサーバかつonnxruntime-gpuがインストールされている環境に限り、`--device cuda`を指定することでGPUを利用した処理に切り替える。


## OCR結果の例


## モデルの再学習及びカスタマイズについて（開発者向け情報）
[学習及びモデル変換手順](/train/README.md)をご覧ください。


## 技術情報について（開発者向け情報）

NDLOCR-Liteは「レイアウト認識」、「文字列認識」、「読み順整序」の3つの機能（モジュール）を組み合わせて実現しています。

レイアウト認識にはDEIMv2[1]、文字列認識にはPARSeq[2]をそれぞれ用いており、読み順整序については当館が公開している[NDLOCR](https://github.com/ndl-lab/ndlocr_cli)と同様のモジュールを用いています。

[1]Shihua Huang and Yongjie Hou and Longfei Liu and Xuanlong Yu and Xi Shen. Real-Time Object Detection Meets DINOv3. arXiv preprint arXiv:2509.20787, 2022.(https://arxiv.org/abs/2509.20787)

[2]Darwin Bautista, Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. arXiv:2212.06966, 2022. (https://arxiv.org/abs/2207.06966)

レイアウト認識及び文字列認識の機械学習モデルは、いずれもpytorchをフレームワークとした学習を行った後にONNX形式に変換して利用しています。詳しくは[学習及びモデル変換手順](/train/README.md)をご覧ください。


