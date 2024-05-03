---
level: 2
layout: center
---

それぞれのモデルの説明をする前に

# 5.5. フォルダ構成

---
level: 2
layout: center
---

<!-- apply style to children -->
````md magic-move { style: '--slidev-code-font-size: 1.1rem; --slidev-code-line-height: 1.5;' }
```bash
parediffusers
├── __init__.py
├── defaults.py
├── models
│   ├── __init__.py
│   ├── attension.py
│   ├── embeddings.py
│   ├── resnet.py
│   ├── transformer.py
│   ├── transformer_blocks.py
│   ├── unet_2d_blocks.py
│   ├── unet_2d_get_blocks.py
│   ├── unet_2d_mid_blocks.py
│   └── vae_blocks.py
├── pipeline.py
├── scheduler.py
├── unet.py
├── utils.py
└── vae.py
```
```bash
parediffusers
├── __init__.py 
├── defaults.py
├── models # UNetやVAEの構築のためのモジュール
│   ├── __init__.py
│   ├── attension.py
│   ├── embeddings.py
│   ├── resnet.py
│   ├── transformer.py
│   ├── transformer_blocks.py
│   ├── unet_2d_blocks.py
│   ├── unet_2d_get_blocks.py
│   ├── unet_2d_mid_blocks.py
│   └── vae_blocks.py
├── pipeline.py # 画像生成のためのパイプライン 5. Pipelineで詳しく説明
├── scheduler.py # DDIMSchedulerの実装 4. Schedulerで詳しく説明
├── unet.py # UNet2DConditionModelの実装 6. UNetで詳しく説明
├── utils.py # 活性化関数などのユーティリティ関数
└── vae.py # AutoencoderKLの実装 8. VAEで詳しく説明
```
````

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />defaults.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/defaults.py)</span>

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fdefaults.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />models/](https://github.com/masaishi/parediffusers/tree/main/src/parediffusers/models)</span>
UNetやVAEの構築のためのモジュール

- `attention.py`: TransformerBlockやUnetで使われるAttentionモジュールの実装
- `embeddings.py`: UNetで使われるTimestepsなどの実装
- `resnet.py`: UNetで使われるResNetなどの実装
- `transformer.py`: UNetで使われるTransformerの実装
- `transformer_blocks.py`: Transformerに使われるTransformerBlockの実装
- `unet_2d_blocks.py`: get_unet_2d_blocksで使われるUNetBlockの実装
- `unet_2d_get_blocks.py`: UNetやVAEのEncoderやDecoderで使われるget_up_blockやget_down_blockの実装
- `unet_2d_mid_blocks.py`: UNetで使われるUNetMidBlockの実装
- `vae_blocks.py`: VAEで使われるVAEBlockの実装

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />pipeline.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/pipeline.py)</span>
実際にText2Imgを行う

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fpipeline.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />scheduler.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/scheduler.py)</span>
6. で詳しく説明

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fscheduler.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />unet.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/unet.py)</span>
7. で詳しく説明

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Funet.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />utils.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/utils.py)</span>
活性化関数を扱う

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Futils.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />vae.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/vae.py)</span>
8. で詳しく説明

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fvae.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>
