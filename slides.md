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
src: /slides/intro.md
---

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

ざっくりしたLDMの説明

<h1>1. PromptをEmbeddingに変換する<br />
2. ランダムなLatentを作る<br />
3. UNetで、デノイジングを行う<br />
4. VAEでデコードし、画像を生成する</h1>

---
level: 2
layout: center
---

実際にText2Imgを行うコード

<iframe frameborder="0" scrolling="no" style="width:45rem; height:163px;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2F035772c684ae8d16c7c908f185f6413b72658126%2Fsrc%2Fparediffusers%2Fpipeline.py%23L131-L134&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: center
---

Latent Diffusion Model (LDM)とは?

# Latent Space (滞在空間)で、
# <span v-mark.yellow="1">DDPM</span>を動かすモデル

---
level: 2
---

Denoising Diffusion Probabilistic Model (DDPM)とは?

<h1 class="!text-7.9">画像に<span v-mark.red="1">ノイズを加え</span>、そこから<span v-mark.blue="2">元の画像に復元</span>するモデル</h1>

<p>音声などのデータ全般に活用されていますが、このスライドでは画像について説明します。</p>

<ul>
	<li><span v-mark.red="1">Diffusion process(拡散過程)</span>を用い、学習データの前処理を行う。確率過程（マルコフ連鎖)</li>
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

# Diffusionは前処理でもNNでもないのに、
# Diffusion Modelと呼ばれるが面白い

---
level: 2
layout: center
---

Latent Diffusion Model (LDM)とは?

# <span v-mark.green="1">Latent Space (滞在空間)</span>で、
# DDPMを動かすモデル

---
level: 2
layout: center
transition: fade
---

目的関数間違い探し

<h2>
$$
L_{DM} := \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(x_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<!--
上がDiffusion Model
下がLatent Diffusion Model

\mathcal  = 筆記体(カリグラフィー)、なんて読めばいい? AIにおいてはEncorderと読んじゃっていい?
\mathbb{E}_{x ではなく、VAE Encorderを通したlatent spaceを期待値の計算に使っている。
-->

---
level: 2
layout: center
transition: fade
---

Latent Diffusion Model (LDM)

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<!--
\mathcal{E} = VAE Encorder
\mathcal{N} = ガウシアンノイズ
\epsilon_\theta = VAE Decoder

E(期待値): 入力データ がEncorderを通してlatent spaceに変換されている。
VAE Decoderを通すとき、t=timestepを考慮している。
-->

---
level: 2
layout: center
transition: fade
---

Latent Diffusion Model (LDM)

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<!--
\mathcal{E} = VAE Encorder
\mathcal{N} = ガウシアンノイズ
\epsilon_\theta = VAE Decoder

E(期待値): 入力データ がEncorderを通してlatent spaceに変換されている。
VAE Decoderを通すとき、t=timestepを考慮している。
-->

---
level: 2
layout: center
---

Latent Diffusion Model (LDM) with Conditioning

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0, 1), t }\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t, \tau_\theta(y)) \Vert_{2}^{2}\Big] \, ,
$$
</h2>

<v-clicks every="1" at="1">

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V
$$

$$
\begin{equation*}
Q = W^{(i)}_Q \cdot  \varphi_i(z_t), \; K = W^{(i)}_K \cdot \tau_\theta(y),
  \; V = W^{(i)}_V \cdot \tau_\theta(y) . \nonumber
%
\end{equation*}
$$

</v-clicks>

<!--
Conditioning、つまりpromptやsemantic map、repres entations, imagesなどを考慮している。
-->

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
		<span class="text-xs ml-27.5% mt-0 mb-0">Transformerの次に死ぬほど目にしたStable Diffusionの図</span>
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

Latent Space(滞在空間)とは?

# 入力画像の特徴を抽出した空間

<!--
TODO: VAEを通した画像の平均をとった画像を用意する。
-->

---
level: 2
layout: center
---

ざっくりした説明

<h1>1. PromptをEmbeddingに変換する<br />
2. ランダムなLatentを作る<br />
3. UNetで、デノイジングを行う<br />
4. VAEでデコードし、画像を生成する</h1>

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
transition: fade
---

# 5.1. encode_prompt

<v-clicks every="1" at="1">

- L54: `CLIPTokenizer`: テキスト(prompt)をトークン化。ベクトルにすることで、AIに扱いやすくさせる。

- L56: `CLIPTextModel`: 言語と画像のマルチモーダルモデル。画像生成においては、プロンプトで作りたい画像の表現（embedding）を抽出する。

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L21-L39](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L21-L39)

```python {4,5|4|4,5}{lines:false,at:1}
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

```python {54,56|54|54,56}{lines:true,startLine:50,at:1}
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
transition: fade
---

# 5.1. encode_prompt


- L54: `CLIPTokenizer`: テキスト(prompt)をトークン化。ベクトルにすることで、AIに扱いやすくさせる。

- L56: `CLIPTextModel`: 言語と画像のマルチモーダルモデル。画像生成においては、プロンプトで作りたい画像の表現（embedding）を抽出する。


<v-clicks every="1" at="2">

- L46: シンプルにするために、negative_promptは空の文字列としています。

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L34-L35](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L34-L35)

```python {all}{lines:true,startLine:34,at:1}
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
```

[<mdi-github-circle />pipeline.py#L41-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L57)

```python {|all|46|all}{lines:true,startLine:41,at:1}
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

<iframe frameborder="0" scrolling="no" class="emg-iframe-text-inputs" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-text_inputs.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>


<style>
	.emg-iframe-text-inputs {
		transform: scale(0.9) translate(-50%, -50%); /* Apply both transformations */
		transform-origin: top left;
		position: absolute;
		top: 50%;
		left: 50%;
		width: 100%;
		height: 100%;
	}
</style>

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="emg-iframe-prompt-embeds" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-prompt-embeds {
		transform: scale(0.8) translate(-50%, -50%); /* Apply both transformations */
		transform-origin: top left;
		position: absolute;
		top: 57%;
		left: 50%;
		width: 100%;
		height: 130%;
	}
</style>

---
level: 2
layout: center
---

<iframe frameborder="0" scrolling="yes" class="overflow-scroll emg-iframe-play-prompt-embeds" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fnotebooks%2Fch0.0.2_Play_prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-play-prompt-embeds {
		transform: scale(0.5) translate(-50%, -50%); /* Apply both transformations */
		transform-origin: top left;
		position: absolute;
		top: 50%;
		left: 50%;
		width: 100%;
		height: 160%;
	}
</style>

<!--
まるでWord2Vec
-->

---
level: 2
layout: custom-two-cols
leftPercent: 0.3
---

# 5.2. get_latent

<v-clicks every="1">

- L63: 1/8のサイズのランダムなテンソルを生成

<img src="/exps/latent.png" class="mt-5 h-48 object-contain" />

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L59-L65](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L59-L65)


```python {all|63|all}{lines:true,startLine:59,at:1,style:'--slidev-code-font-size: 1rem; --slidev-code-line-height: 1.5;'}
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
leftPercent: 0.5
transition: fade
---

# 5.3. denoise

<v-clicks every="1">

- L86: UNet

- L91: Scheduler

</v-clicks>

::right::


[<mdi-github-circle />pipeline.py#L75-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L75-L93)

```python {all|86|86,91}{lines:true,startLine:75,at:1}
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
layout: custom-two-cols
leftPercent: 0.5
transition: fade
---

# 5.3. denoise

- L86: UNet2DConditionModel

- L91: DDIMScheduler


::right::

[<mdi-github-circle />pipeline.py#L21-L39](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L21-L39)

```python {6,7}{lines:false,at:1}
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

[<mdi-github-circle />pipeline.py#L82-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L82-L93)

```python {86,91}{lines:true,startLine:82,at:1}
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
layout: custom-two-cols
leftPercent: 0.5
---

# 5.3. denoise

<v-clicks every="1">

- L80: Schedulerを使いtimestepsの取得 <br />(<span class="text-sm">7. Schedulerで詳しく説明</span>)

- L82: timestepsの長さ分ループ<br />(<span class="text-sm">timestepsの長さ分 = num_inference_steps</span>)

- L86: UNetでデノイズ <br />(<span class="text-sm">8. UNetで詳しく説明</span>)

- L88: どれだけプロンプトを考慮するかを計算 <br />(<span class="text-3">参考: 
Jonathan Ho, Tim Salimans: “Classifier-Free Diffusion Guidance”, 2022; <a href='http://arxiv.org/abs/2207.12598'>arXiv:2207.12598</a>.</span>)

- L91: Schedulerによって、デノイズの強さを決定

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L82-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L82-L93)

```python {all|80|82|86|88|91|all}{lines:true,startLine:75,at:1}
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
layout: center
---

# 5.3. denoise
SchedulerとUNetを使うということだけ覚えておいてください。

::right::

[<mdi-github-circle />pipeline.py#L82-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L82-L93)

```python {all|80|82|86|88|91|all}{lines:true,startLine:75,at:1}
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
layout: custom-two-cols
leftPercent: 0.4
---

# 5.4. vae_decode

<v-clicks every="1">
</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L107-L105](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L107-L115)

```python {all}{lines:true,startLine:107,at:1}
@torch.no_grad()
def vae_decode(self, latents):
	"""
	Decode the latent tensors using the VAE to produce an image.
	"""
	image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
	image = self.denormalize(image)
	image = self.tensor_to_image(image)
	return image
```

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

# <span class="text-3xl">[<mdi-github-circle />scheduler.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/scheduler.py)</span>
デノイズの強さを決定

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fscheduler.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

pipe.scheduler.step(guided_noise_residual, t, latents) をうまく使って、schedulerがどんなことをしているかのアニメーションを作りたい。

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

# [<mdi-github-circle />unet.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/unet.py)
デノイジングに使われる

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Funet.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: image
image: /images/unet-figure.png
backgroundSize: 70%
class: 'text-black'
---

<!-- Reference -->
<p class="text-black text-xs abs-bl w-full mb-6 text-center">
Olaf Ronneberger, Philipp Fischer, Thomas Brox: “U-Net: Convolutional Networks for Biomedical Image Segmentation”, 2015; <a href='http://arxiv.org/abs/1505.04597'>arXiv:1505.04597</a>.
</p>

---
level: 2
layout: center
---

# 7.1 モデル作成

```python
class PareUNet2DConditionModel(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.config = DotDict(DEFAULT_UNET_CONFIG)
		self.config.update(kwargs)
		self.config.only_cross_attention = [self.config.only_cross_attention] * len(self.config.down_block_types)
		self.config.num_attention_heads = self.config.num_attention_heads or self.config.attention_head_dim
		self._setup_model_parameters()

		self._build_input_layers()
		self._build_time_embedding()
		self._build_down_blocks()
		self._build_mid_block()
		self._build_up_blocks()
		self._build_output_layers()
```

---
level: 2
---

Transformer使っていること書く?

---
level: 2
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
level: 2
layout: center
---

# Variational Autoencoder
変分自己符号化器 (日本語訳かっこいい!)

---
level: 2
---

# [<mdi-github-circle />vae.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/vae.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fvae.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
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

---
level: 2
layout: center
---

# ライブラリのコードを読むの楽しい!

---
level: 2
layout: center
---

## ライブラリの至る所で、論文が引用

[<mdi-github-circle />diffusers/.../pipeline.py](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py)

<img src="/images/diffusers-code-arxiv.png" class="mt-5 h-92 object-contain" />

---
level: 2
layout: center
---

まとめ
# 1. プロンプトのエンコード
# 2. ランダムな潜在空間の生成
# 3. UNetを用いてデノイジング
# 4. Latent SpaceからPixel Spaceへのデコード

---
level: 2
layout: center
---

# ご清聴ありがとうございました！
