# TensorProjection Layer：台灣華語語音辨識範例

本 repository 收錄中國統計學社投稿論文中，TensorProjection Layer（TPL）應用於台灣華語語音辨識範例所使用的程式碼。

## 資料

本分析使用 Mozilla Common Voice 資料集中的**台灣華語**資料。分析僅使用 `validated.tsv` 中的資料，並在前處理時移除長度超過 5 秒的錄音。論文中的分析經過上述前處理後，共保留 **81,249 筆錄音**。所得到的輸入大小為

```text
(81249, 1, 64, 626)
```

本 repository 不包含原始 Common Voice 資料。

請將資料放置於以下位置：

```text
jcsa2027-TPL/
├── data/
│   ├── validated.tsv
│   └── clips/
├── npy/                  # 由前處理產生，不納入 Git
├── results/              # 由實驗產生，不納入 Git
└── python/
```

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

模型種類與隨機種子可透過環境變數指定。由 `python/` 目錄執行：

```bash
MODEL=baseline SEED=123 jupyter nbconvert \
  --execute --to notebook experiment.ipynb \
  --output /tmp/experiment_baseline_seed123.ipynb
```

或

```bash
MODEL=tpl SEED=123 jupyter nbconvert \
  --execute --to notebook experiment.ipynb \
  --output /tmp/experiment_tpl_seed123.ipynb
```

`experiment_config.py` 中主要的模型訓練設定如下：

- 訓練 / 驗證 / 測試資料分割：85% / 5% / 10%
- 最佳化方法：AdamW
- 學習率：`1e-3`
- weight decay：`1e-4`
- batch size：256
- 最大 epoch 數：100
- early stopping patience：5
- BiGRU hidden size：每個方向 256
- BiGRU 層數：2
- dropout：0.2

每次實驗的結果會儲存於

```text
results/<model>/seed_<seed>/
```

其中包含該次實驗的設定、訓練過程、測試指標以及最佳模型的參數。

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


