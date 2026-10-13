# TensorProjection Layer：台灣華語語音辨識範例

本 repository 收錄中國統計學社投稿論文中，TensorProjection Layer（TPL）應用於台灣華語語音辨識範例所使用的程式碼。

## 資料

請下載 Mozilla Common Voice 的台灣華語資料，並依照以下結構放置檔案：

```text
jcsa2027-TPL/
├── data/
│   ├── validated.tsv
│   └── clips/
├── npy/                  # 前處理後自動產生
├── results/              # 執行實驗後自動產生
└── python/
```

原始資料不包含在本 repository 中。`npy/` 與 `results/` 無須事先建立。

## Repository 結構

```text
python/
├── step1_preprocess.ipynb
├── step2_translate_to_bopomofo.ipynb
├── step3_build_syllable_vocab.ipynb
├── experiment.ipynb
├── experiment_config.py
├── run_20x.sh
├── analyze_20x2_results.ipynb
└── py/
    ├── bopomofo_utils.py
    ├── evaluation.py
    ├── loss.py
    ├── metrics.py
    ├── models.py
    ├── taiwan_bopomofo_equiv.py
    ├── tensor_projection.py
    └── train_utils.py
```

## 環境設定

建立 Python 環境後，安裝所需套件：
```bash
pip install -r requirements.txt
```

## 重現資料前處理

請由 `python/` 目錄依序執行以下 notebook：

```text
step1_preprocess.ipynb
step2_translate_to_bopomofo.ipynb
step3_build_syllable_vocab.ipynb
```

前處理完成後，會在 `../npy/` 產生以下檔案：

```text
npy/
├── X.npy
├── meta.csv
└── vocab.json
```

## 執行單次實驗

開啟 `python/experiment_config.py`，設定模型種類（`MODEL`）與隨機種子（`SEED`）。接著使用 Jupyter Lab 開啟 `python/experiment.ipynb`，依序執行所有 cell，即可進行模型訓練與評估。其他訓練設定也可以在 `experiment_config.py` 中調整。實驗結果會儲存於 `results/<model>/seed_<seed>/`。

## 重現 20 × 2 次比較

論文中的比較對兩個模型皆使用 random seed 120--139，共分別執行 20 次。

執行：

```bash
cd python
bash run_20x.sh
```

即可分別執行 20 次 Baseline model 與 20 次 TPL model。

所有實驗完成後，執行

```text
analyze_20x2_results.ipynb
```

即可彙整測試結果，並繪製 CTC loss、syllable accuracy 與 exact accuracy 三項指標的 boxplot。


