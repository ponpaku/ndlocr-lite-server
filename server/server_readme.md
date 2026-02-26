# NDLOCR-Lite OCR Server

NDLOCR-Lite のモデルを使って画像・PDF から日本語テキストを抽出する、ローカル動作の Web OCR サーバーです。

---

## 目次

1. [概要](#概要)
2. [ファイル構成](#ファイル構成)
3. [必要環境](#必要環境)
4. [セットアップ](#セットアップ)
5. [起動](#起動)
6. [Web UI の使い方](#web-ui-の使い方)
7. [API リファレンス](#api-リファレンス)
8. [環境変数・設定](#環境変数設定)
9. [CUDA / GPU 利用](#cuda--gpu-利用)
10. [アーキテクチャメモ](#アーキテクチャメモ)

---

## 概要

| 項目 | 内容 |
|---|---|
| フレームワーク | FastAPI + Uvicorn |
| デフォルト URL | `http://127.0.0.1:7860` |
| OCR エンジン | NDLOCR-Lite（DEIM 検出 + PARSEQ 認識） |
| 対応入力 | 画像（JPEG / PNG / BMP / TIFF など）、PDF |
| PDF 処理 | pypdfium2 でページ画像変換後に OCR |
| 並列処理 | PDF 複数ページを `MAX_PAGE_WORKERS` 枚同時処理 |

---

## ファイル構成

```
server/
├── main.py               # FastAPI アプリ本体
├── server_readme.md      # このファイル
├── templates/
│   └── index.html        # Web UI（SPA、ビルド不要）
└── static/
    └── style.css         # 旧テンプレート用（現在は index.html に CSS 埋め込み済み）
```

モデルファイルは別途 `src/model/` に配置が必要です（後述）。

---

## 必要環境

### Python パッケージ

`requirements.txt` または `pyproject.toml` に記載されている依存関係を参照してください。
サーバー動作に必須のものは以下です。

| パッケージ | 用途 |
|---|---|
| `fastapi` | Web フレームワーク |
| `uvicorn` | ASGI サーバー |
| `onnxruntime-gpu` | OCR モデル推論（CPU でも動作） |
| `pypdfium2` | PDF → 画像変換 |
| `pillow` | 画像処理 |
| `numpy` | 配列処理 |
| `pyyaml` | 設定ファイル読み込み |

### モデルファイル

以下のファイルをリポジトリルートの `src/model/` に配置してください。
モデルが 1 つでも欠けている場合、サーバーは**起動を拒否**します。

```
src/model/
├── deim-s-1024x1024.onnx
├── parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx
├── parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx
└── parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx
```

---

## セットアップ

```bash
# リポジトリルートで
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 起動

```bash
# リポジトリルートから実行
python server/main.py
```

起動成功時のログ例：

```
[CUDA] registered: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin
[startup] device=cuda, MAX_PAGE_WORKERS=2
[NDLOCREngine] Loading PARSEQ recognizers …
[NDLOCREngine] Loading 2 DEIM detector(s) …
[NDLOCREngine] Ready.
INFO:     Uvicorn running on http://127.0.0.1:7860
```

ブラウザで `http://127.0.0.1:7860` を開くと Web UI が表示されます。

---

## Web UI の使い方

1. ファイルをドラッグ＆ドロップ、またはクリックして選択（画像・PDF 対応）
2. **詳細設定**（任意）
   - **改行処理** : OCR 後のテキスト整形モード
     - `処理しない` : 認識結果をそのまま出力
     - `段落整形` : 段落間の空行を維持しつつ行内改行を結合
     - `強力整形` : 改行をほぼ除去してベタ書きに
   - **dpi** : PDF のラスタライズ解像度（デフォルト 220）。高いほど精細だが処理時間増
   - **device** : `auto`（推奨）/ `cuda` / `cpu`
   - **reading_order** : 読み順の指定（`auto` / `ltr_ttb` / `rtl_ttb`）
3. **実行** ボタンを押すと処理開始
4. 結果パネルにページごとのテキストが表示される
   - 複数ページの場合はページセレクターで切り替え
   - **コピー** ボタンでクリップボードに一括コピー
5. **中断** ボタンで処理中断可能（PDF の次ページ以降をスキップ）

---

## API リファレンス

### `GET /api/status`

サーバーのモデル・デバイス情報を返します。

**レスポンス例**
```json
{
  "model": "NDLOCR-Lite",
  "device_default": "cuda",
  "cuda_available": true
}
```

---

### `POST /api/analyze`

ファイルをアップロードして OCR を実行します。

**リクエスト** : `multipart/form-data`

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `file` | file | **必須** | 画像または PDF |
| `dpi` | int | `220` | PDF ラスタライズ解像度 |
| `linebreak_mode` | string | `none` | `none` / `paragraph` / `compact` |
| `reading_order` | string | `auto` | `auto` / `ltr_ttb` / `rtl_ttb` |
| `device` | string | `auto` | `auto` / `cuda` / `cpu` ※起動時に固定済み |
| `request_id` | string | `""` | 進捗ポーリング用 UUID（クライアントが生成） |

**レスポンス例**
```json
{
  "results": [
    {
      "page": 1,
      "text": "認識されたテキスト",
      "raw": "認識されたテキスト（整形前）",
      "json": null,
      "error": null,
      "layout_preview_base64": null,
      "blocks": []
    }
  ],
  "task": "text",
  "linebreak_mode": "none",
  "page_count": 1,
  "device": "cuda",
  "reading_order": "auto",
  "state": "done"
}
```

`state` は `done` / `canceled` / `error` のいずれか。

---

### `GET /api/progress/{request_id}`

処理中の進捗を返します。`/api/analyze` 実行中に 500ms 間隔でポーリングします。

**レスポンス例**
```json
{
  "state": "processing",
  "message": "ページ 2/5 完了...",
  "total_pages": 5
}
```

`state` は `processing` / `canceling` / `done` / `error` / `canceled` のいずれか。

---

### `POST /api/cancel/{request_id}`

処理中断を要求します。PDF の現在処理中ページが完了した後に中断されます。

**レスポンス例**
```json
{
  "message": "中断要求を受け付けました"
}
```

---

## 環境変数・設定

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `NDLOCR_PAGE_WORKERS` | `2` | PDF の同時処理ページ数。増やすと速くなるが VRAM / RAM 消費が増える |
| `CUDNN_PATH` | - | cuDNN の bin ディレクトリパス。自動検出できない場合に設定 |
| `CUDA_PATH` | - | CUDA Toolkit のルートパス |

---

## CUDA / GPU 利用

### 動作要件

| ソフトウェア | 要件 |
|---|---|
| CUDA Toolkit | 12.x |
| cuDNN | **9.x**（8.x では動作しない） |
| onnxruntime-gpu | 1.23.2 |

### 起動時の自動判定ロジック

1. `CUDNN_PATH` / `CUDA_PATH` 環境変数および標準インストールパスから CUDA / cuDNN の bin ディレクトリを探索し、プロセスの DLL 検索パスに追加
2. `cublasLt64_12.dll`（cuBLAS）と `cudnn64_9.dll`（cuDNN）が読み込めるかを確認
3. 両方利用可能な場合のみ `device=cuda` で起動、それ以外は `device=cpu` にフォールバック

### よくあるエラーと対処

| ログ | 原因 | 対処 |
|---|---|---|
| `cudnn64_9.dll not found` | cuDNN 9.x 未インストール | NVIDIA Developer から cuDNN 9.x をインストール |
| `cublasLt64_12.dll not found` | CUDA 12.x Toolkit 未インストール、またはパス未登録 | CUDA 12.x をインストールし `CUDA_PATH` を設定 |
| cudnn DLL が `cudnn64_8.dll` と表示される | cuDNN 8.x がインストールされている | cuDNN 9.x に更新 |

---

## アーキテクチャメモ

### モデルキャッシュ（`NDLOCREngine`）

サーバー起動時にモデルを一度だけロードし、以降のリクエストで使い回します。
ページごとにモデルをロードしていた場合と比べ、2 ページ目以降の処理時間が大幅に短縮されます。

### スレッド安全性

| モデル | スレッド安全 | 理由 | 実装 |
|---|---|---|---|
| PARSEQ（認識器 × 3） | ✅ | `onnxruntime.InferenceSession.run()` はスレッドセーフ | 全ワーカーで共有 |
| DEIM（検出器） | ❌ | `preprocess()` が `self.image_width/height` を書き換える | `queue.Queue` でプール管理 |

### PDF 並列処理

```
PDF Nページ
  └─ infer_pages()
       ├─ page 1 ──→ DEIM pool[0] → 共有 PARSEQ
       ├─ page 2 ──→ DEIM pool[1] → 共有 PARSEQ
       ├─ page 3 ──→ (pool[0] 空き待ち)
       └─ ...
```

`MAX_PAGE_WORKERS`（デフォルト 2）= DEIM インスタンス数 = 同時処理ページ数
