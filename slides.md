---
# try also 'default' to start simple
theme: seriph
colorSchema: 'light'

# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: backgrounds/understand-sd.png
# some information about your slides, markdown enabled
title: ソースコードから理解するStable Diffusion
info: |
  ## Stable Diffusionの画像生成コードの解説スライド
  diffusersをシンプル化した、parediffusersというライブラリのコードを通じて、Latent Diffusion Modelsの画像生成の仕組みを解説するスライドです。

author: masaishi
keywords: [ Stable Diffusion, Diffusers, parediffusers, AI, ML, Generative Models ]

export:
  format: pdf
  timeout: 30000
  dark: false
  withClicks: false

lineNumbers: true

# apply any unocss classes to the current slide
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: true
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true

fonts:
  sans: Noto Serif JP, serif
---

# ソースコードから理解するStable Diffusion

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Understand Stable Diffusion from code, cyberpunk theme, best quality, high resolution, concept art</p>

---
title: 自己紹介
---

# 2. 石原 正宗 (Masamune Ishihara)
<div class="[&>*]:important-leading-10 opacity-80">
Computer Engineering Undergrad at University of California, Santa Cruz <br />
AI/MLとGISに興味があります。 <br />

<br />

#### 好きなもの:
- 紅茶
- テニス
- Rebuild.fm (<a href="https://rebuild.fm/223/" target="_blank" class="ml-1.5 border-none!">223: Ear Bleeding Pods (higepon)</a>を聞いてkaggleを始めました。)
</div>

<div class="mt-10 flex flex-col gap-2">
  <div>
		<mdi-github-circle />
		<a href="https://github.com/masaishi" target="_blank" class="ml-1.5 border-none! font-300">masaishi</a>
	</div>
	<div>
		<mdi-twitter />
		<a href="https://twitter.com/masaishi2001" target="_blank" class="ml-1.5 border-none! font-300">@masaishi2001</a>
	</div>
	<div>
		<mdi-linkedin />
		<a href="https://www.linkedin.com/in/masamune-ishihara" target="_blank" class="ml-1.5 border-none! font-300">masamune-ishihara</a>
	</div>
	<div class="flex items-center">
		<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512" class="h-5 w-5"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M304.2 501.5L158.4 320.3 298.2 185c2.6-2.7 1.7-10.5-5.3-10.5h-69.2c-3.5 0-7 1.8-10.5 5.3L80.9 313.5V7.5q0-7.5-7.5-7.5H21.5Q14 0 14 7.5v497q0 7.5 7.5 7.5h51.9q7.5 0 7.5-7.5v-109l30.8-29.3 110.5 140.6c3 3.5 6.5 5.3 10.5 5.3h66.9q5.3 0 6-3z"/></svg>
		<a href="https://www.kaggle.com/masaishi" target="_blank" class="ml-1.5 border-none! font-300">masaishi</a>
	</div>
</div>

<img src="/images/icon_tea_light.png" class="rounded-full w-35 abs-tr mt-12 mr-24" />

---
level: 2
layout: center
---

# Kagglerにとって画像生成を使う機会は少ない?

<img src="/images/stable-diffusion-image-to-prompts.png" class="h-100" />

<a src="https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/overview" target="_blank" class="abs-bl w-full mb-6 text-center text-xs text-black border-none!">https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/overview</a>

---
level: 2
layout: center
---

このスライドの目的

# 画像生成のコードを1通り紹介したい。

---
level: 2
layout: center
---

# このスライドについて
スライドを書くにあたって、自分が理解できていなかったところなどを多く実感しました。間違った説明をしているかもしれないので、不明瞭なところや、間違いを下記のリンクから教えていただけると幸いです。

<br />

### [<mdi-github-circle />understand-stable-diffusion-slidev-ja](https://github.com/masaishi/understand-stable-diffusion-slidev-ja)

<br />

[<mdi-radiobox-marked />Issues](https://github.com/masaishi/understand-stable-diffusion-slidev-ja/issues): 間違いを見つけたらご指摘ください。
<br />

[<mdi-message-text-outline />Discussions](https://github.com/masaishi/understand-stable-diffusion-slidev-ja/discussions): 質問があればこちらからお願いします。
<br />

[<mdi-source-pull />Pull Requests](https://github.com/masaishi/understand-stable-diffusion-slidev-ja/pulls): もし修正があればお送りください。
<br />

---
layout: center
title: 目次
---

# 目次
<Toc minDepth="1" maxDepth="1"></Toc>

---
layout: cover
title: Stable Diffusionの概要
background: /backgrounds/stable-diffusion.png
---

# 4. Stable Diffusionの概要

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Stable Diffusion, watercolor painting, best quality, high resolution</p>

---
level: 2
layout: center
---

# Stable Diffusionとは?

- StabilityAIによって開発されたLatent Diffusion Model (LDM)をベースとした画像生成モデル
- Text-to-Image, Image-to-Imageなどのタスクに利用可能
- Diffusersを利用することで簡単に動かすことができる。
- https://arxiv.org/abs/2112.10752

---
level: 2
layout: center
---

Latent Diffusion Model (LDM)とは?

# Denoising Diffusion Probabilistic Model (DDPM) に Latent Space(滞在空間)という概念を追加した仕組み

---
level: 2
---

Denoising Diffusion Probabilistic Model (DDPM)とは?

<h1 class="!text-7.9">画像に<span v-mark.red="1">ノイズを加え</span>、そこから<span v-mark.blue="2">元の画像に復元</span>するモデル</h1>

<p>音声などのデータ全般に活用されていますが、このスライドでは画像について説明します。</p>

<ul>
	<li><span v-mark.red="1">Diffusion process(拡散過程)</span>を用い、学習データの前処理を行う。確率過程（特にマルコフ連鎖)</li>
	<li><span v-mark.blue="2">Reverse process(逆拡散過程)</span>を用い、ノイズを加えられたデータから元のデータを復元する。</li>
</ul>

<br />

<img src="/images/ddpm-figure.png" class="abs-b mb-10 ml-auto mr-auto w-5/6" />

<!-- Reference -->
<p class="text-black text-xs abs-bl w-full mb-6 text-center">
Jonathan Ho, Ajay Jain, Pieter Abbeel: “Denoising Diffusion Probabilistic Models”, 2020; <a href='http://arxiv.org/abs/2006.11239'>arXiv:2006.11239</a>.
</p>

---
level: 2
layout: center
---

Latent Diffusion Model (LDM)とは?

# Latent(滞在)空間で、Denoising Diffusion Probabilistic Model (DDPM) を計算する仕組み

---
level: 2
---

<div class="flex flex-col !justify-between w-full h-120">
	<div>
		<img src="/images/ddpm-figure.png" class="ml-auto mr-auto h-26" />
		<!-- Reference -->
		<p class="text-black text-xs w-full mt-6 text-center">
		Jonathan Ho, Ajay Jain, Pieter Abbeel: “Denoising Diffusion Probabilistic Models”, 2020; <a href='http://arxiv.org/abs/2006.11239'>arXiv:2006.11239</a>.
		</p>
	</div>
	<div v-click>
		<img src="/images/stable-diffusion-figure.png" alt="Stable Diffusion Figure" class="ml-auto mr-auto h-48 object-contain" />
		<p class="text-black text-xs w-full mt-6 text-center">
		Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer: “High-Resolution Image Synthesis with Latent Diffusion Models”, 2021; <a href='http://arxiv.org/abs/2112.10752'>arXiv:2112.10752</a>.
		</p>
	</div>
</div>

---
level: 2
layout: center
---

Latent Diffusion Model (LDM)とは?

# Latent(滞在)空間で、Denoising Diffusion Probabilistic Model (DDPM) を計算する仕組み

---
level: 2
layout: center
---

Latent Space(滞在空間)とは?

# 入力画像の特徴を抽出した空間 

TODO: VAEを通した画像の平均をとった画像を用意する。

---
level: 2
layout: center
---

ざっくりした説明

<h1>ランダムなLatentを作る<br />
UNetで、デノイジングを行う<br />
VAEでデコードし、画像を生成する</h1>

---
level: 2
layout: center
---

# Diffusersとは?

- Hugging Face🤗によって開発されたDiffusion Modelsを扱うライブラリ
- 画像生成モデルを簡単に動かすことができる。
- <mdi-github-circle /> https://github.com/huggingface/diffusers

---
level: 2
layout: image-right
image: /exps/d-sd2-sample-42.png
---

# [<mdi-github-circle />Diffusers](https://github.com/huggingface/diffusers)を試す
## <!-- TODO: Find better way, currently for avoide below becomes subtitle -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EbqeoWL5kPaDA8INLWl8g34v3vn83AQ5?usp=sharing)

Install the Diffusers library:
```python
!pip install transformers diffusers accelerate -U
```

Generate an image from text:
```python {all|4-7|8|10|all}{lines:true}
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2",
  dtype=torch.float16,
).to(device=torch.device("cuda"))
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"

image = pipe(prompt, width=512, height=512).images[0]
display(image)
```

---
level: 2
layout: center
---

# Diffusersは機能が豊富で柔軟性も高いが、<br />
# その分コードの理解に時間がかかる。

---
level: 2
---

# [<mdi-github-circle />diffusers/.../pipeline_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll mt-10" style="width:100%; height:85%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fhuggingface%2Fdiffusers%2Fblob%2Fmain%2Fsrc%2Fdiffusers%2Fpipelines%2Fstable_diffusion%2Fpipeline_stable_diffusion.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# [<mdi-github-circle />parediffusers/.../pipeline.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/pipeline.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll mt-10" style="width:100%; height:85%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fpipeline.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: image-right
image: /exps/p-sd2-sample-43.png
---

# [<mdi-github-circle />PareDiffusers](https://github.com/masaishi/parediffusers)
## <!-- TODO: Find better way, currently for avoide below becomes subtitle -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I-qU3hfF19T42ksIh5FC0ReyKZ2hsJvx?usp=sharing)

Install the PareDiffusers library:
```python
!pip install parediffusers
```

Generate an image from text:
```python {all|2|4|6}{lines:true}
import torch
from parediffusers import PareDiffusionPipeline

pipe = PareDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2",
  device=torch.device("cuda"),
  dtype=torch.float16,
)
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"

image = pipe(prompt, width=512, height=512)
display(image)
```

---
layout: cover
title: Pipeline
background: /backgrounds/pipeline.png
---

# 5. Pipeline

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Pipeline, cyberpunk theme, best quality, high resolution, concept art</p>

---
level: 2
layout: image
image: /images/stable-diffusion-figure.png
backgroundSize: 70%
class: 'text-black'
---

<!-- Reference -->
<p class="text-black text-xs abs-bl w-full mb-6 text-center">
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer: “High-Resolution Image Synthesis with Latent Diffusion Models”, 2021; <a href='http://arxiv.org/abs/2112.10752'>arXiv:2112.10752</a>.
</p>

---
level: 2
layout: center
---

<iframe frameborder="0" scrolling="no" style="width:100%; height:163px;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2F035772c684ae8d16c7c908f185f6413b72658126%2Fsrc%2Fparediffusers%2Fpipeline.py%23L131-L134&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<div class="w-full flex flex-col justify-center mt-10">
<img src="/images/stable-diffusion-figure.png" alt="Stable Diffusion Figure" class="h-48 object-contain" />
<p class="text-black text-xs w-full mt-6 text-center">
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer: “High-Resolution Image Synthesis with Latent Diffusion Models”, 2021; <a href='http://arxiv.org/abs/2112.10752'>arXiv:2112.10752</a>.
</p>
</div>

---
level: 2
layout: center
---

````md magic-move
```python {all}{lines:true}
prompt_embeds = self.encode_prompt(prompt)
latents = self.get_latent(width, height).unsqueeze(dim=0)
latents = self.denoise(latents, prompt_embeds, num_inference_steps, guidance_scale)
image = self.vae_decode(latents)
```
```md {all|1|2|3|4|all}
1. `encode_prompt` : テキストのプロンプトをembeddingに変換する。
2. `get_latent` : 生成したい画像サイズの、1/8のスケールでランダムなテンソルを生成する。
3. `denoise` : エンコードされたプロンプトのembeddingから、潜在空間を反復的にデノイズする。
4. `vae_decode` : デノイズされた潜在空間を画像にデコードする。
```
````

---
level: 2
layout: two-cols
transition: fade
---

# 5.1. encode_prompt

::right::

[<mdi-github-circle />pipeline.py#L41-L48](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L48)

```python {all|45|45-46}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
```

---
level: 2
layout: two-cols
transition: fade
---

# 5.1. encode_prompt

::right::

[<mdi-github-circle />pipeline.py#L41-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L57)

```python {all|54|54,56}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
 
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
layout: two-cols
---

# 5.1. encode_prompt

<v-clicks every="1" at="1">

- L34: `CLIPTokenizer`: テキスト(prompt)をトークン化。ベクトルにすることで、AIに扱いやすくさせる。

- L35: `CLIPTextModel`: 言語と画像のマルチモーダルモデル。画像生成においては、プロンプトで作りたい画像の表現（embedding）を抽出する。

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L21-L39](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L21-L39)

```python {all|4|5|4,5}{lines:false,at:1}
@classmethod
def from_pretrained(cls, model_name, device=torch.device("cuda"), dtype=torch.float16):
	# Ommit comments
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
	scheduler = PareDDIMScheduler.from_config(model_name, subfolder="scheduler")
	unet = PareUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
	vae = PareAutoencoderKL.from_pretrained(model_name, subfolder="vae")
	return cls(tokenizer, text_encoder, scheduler, unet, vae, device, dtype)
```

[<mdi-github-circle />pipeline.py#L50-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L50-L57)

```python {all|54|56|54,56}{lines:true,startLine:50,at:1}
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="emg-iframe-text-inputs" style="width:90%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-text_inputs.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-text-inputs {
		/* apply the transform */
		/*--scale-factor: 0.1;*/
		-webkit-transform:scale(0.6);
		-moz-transform:scale(0.6);
		-o-transform:scale(0.6);
		transform:scale(0.6);
		/* position it, as if it was the original size */
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
	}
</style>

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="emg-iframe-prompt-embeds" style="width:80%; height:110%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-prompt-embeds {
		-ms-zoom: 0.65;
		-moz-transform: scale(0.65);
		-moz-transform-origin: 0 0;
		-o-transform: scale(0.65);
		-o-transform-origin: 0 0;
		-webkit-transform: scale(0.65);
		-webkit-transform-origin: 0 0;

		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
	}
</style>

---
level: 2
layout: two-cols
---

# 5.1. encode_prompt

<v-clicks every="1" at="1">

- L45: `get_embes`関数を呼びprompt_embedsを取得

- L46: `get_embes`関数を呼びnegative_prompt_embedsを取得 (シンプルにするために、negative_promptは空の文字列としています。)

- L54: CLIPTokenizerでTokenize

- L56: CLIPTextModelでembeddingを取得

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L34-L35](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L34-L35)

```python {all|none|none|34|35|all}{lines:true,startLine:34,at:1}
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
```

[<mdi-github-circle />pipeline.py#L41-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L57)

```python {all|45|46|54|56|all}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
 
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
---

<iframe frameborder="0" scrolling="yes" class="overflow-scroll mt-10" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fnotebooks%2Fch0.0.2_Play_prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: custom-two-cols
leftPercent: 0.4
---

# 5.2. get_latent

<v-clicks every="1">

- L63: 1/8のサイズのランダムなテンソルを生成

<img src="/exps/latent.png" class="mt-5 h-48 object-contain" />

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L59-L65](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L59-L65)


```python {all|63|all}{lines:true,startLine:59,at:1}
def get_latent(self, width: int, height: int):
	"""
	Generate a random initial latent tensor to start the diffusion process.
	"""
	return torch.randn((4, width // 8, height // 8)).to(
		device=self.device, dtype=self.dtype
	)
```

---
level: 2
layout: custom-two-cols
leftPercent: 0.4
---

# 5.3. denoise

::right::

[<mdi-github-circle />pipeline.py#L75-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L75-L93)

```python {all}
@torch.no_grad()
def denoise(self, latents, prompt_embeds, num_inference_steps=50, guidance_scale=7.5):
	"""
	Iteratively denoise the latent space using the diffusion model to produce an image.
	"""
	timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps)

	for t in timesteps:
		latent_model_input = torch.cat([latents] * 2)
		
		# Predict the noise residual for the current timestep
		noise_residual = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
		uncond_residual, text_cond_residual = noise_residual.chunk(2)
		guided_noise_residual = uncond_residual + guidance_scale * (text_cond_residual - uncond_residual)

		# Update latents by reversing the diffusion process for the current timestep
		latents = self.scheduler.step(guided_noise_residual, t, latents)[0]

	return latents
```

---
level: 2
layout: two-cols
---

# 5.4. vae_decode

---
level: 2
---

# フォルダ構成

````md magic-move
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

# defaults.py

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fdefaults.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# models/

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

# pipeline.py

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fpipeline.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# scheduler.py (6. で詳しく説明)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fscheduler.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# unet.py (7. で詳しく説明)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Funet.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# utils.py

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Futils.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# vae.py (8. で詳しく説明)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll" style="width:100%; height:90%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fvae.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
layout: cover
title: Scheduler
background: /backgrounds/scheduler.png
---

# 6. Scheduler

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Scheduler, flat vector illustration, best quality, high resolution</p>

---
level: 2
---

pipe.scheduler.step(guided_noise_residual, t, latents) をうまく使って、schedulerがどんなことをしているかアニメーションを作りたい。

---
level: 2
layout: image-right
image: /exps/skip-scheduler-result.png
---

5個飛ばしでも画像は生成できるのか?

```python
timesteps, num_inference_steps = pare_pipe.retrieve_timesteps(50)
timesteps = timesteps[::5]

guidance_scale = 7.5

for i, t in enumerate(timesteps):
	latent_model_input = torch.cat([latents] * 2)
	
	# Predict the noise residual for the current timestep
	noise_residual = pare_pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]
	uncond_residual, text_cond_residual = noise_residual.chunk(2)
	guided_noise_residual = uncond_residual + guidance_scale * (text_cond_residual - uncond_residual)

	# Update latents by reversing the diffusion process for the current timestep
	latents = pare_pipe.scheduler.step(guided_noise_residual, t, latents)[0]

image = pare_pipe.vae_decode(latents)
display(image)
```

---
layout: cover
title: UNet
background: /backgrounds/unet.png
---

# 7. UNet

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: UNet, watercolor painting, detailed, brush strokes, best quality, high resolution</p>

---
level: 2
---

UNetの構造の図

---

Transformer使っていること書く?

---

UNetで滞在空間を作って、平均を取れば特徴が抽出できるアニメーションでも作る?

---
layout: cover
title: VAE
background: /backgrounds/vae.png
---

# 8. VAE

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: VAE, abstract style, highly detailed, colors and shapes</p>

---

VAEを変えても画像が生成できる話?
画像生成推論サーバーのVAEを変更機能追加は、インターンで最初にやったことなのでちょっと話せるかも。

---
layout: cover
title: まとめ
background: /backgrounds/summary.png
---

# 9. まとめ

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Summary, long-exposure photography, masterpieces</p>
