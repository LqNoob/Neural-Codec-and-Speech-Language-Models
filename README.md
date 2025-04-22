
<!--
![Neural-Codec-and-Speech-Language-Models](https://socialify.git.ci/LqNoob/Neural-Codec-and-Speech-Language-Models/image?description=1&font=Jost&logo=https%3A%2F%2Fuser-images.githubusercontent.com%2F74038190%2F215282743-5a6fb12c-b67c-45b4-b547-1c340958c6da.png&name=1&pattern=Solid&theme=Light)

<h2 align="center"> Neural Codec and Speech Language Models
</h2>
-->
<h2 align=center> 
<img src="https://socialify.git.ci/LqNoob/Neural-Codec-and-Speech-Language-Models/image?description=1&font=Jost&logo=https%3A%2F%2Fuser-images.githubusercontent.com%2F74038190%2F215282743-5a6fb12c-b67c-45b4-b547-1c340958c6da.png&name=1&pattern=Solid&theme=Light" alt="描述性文字">
</h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLqNoob%2FNeural-Codec-and-Speech-Language-Models&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models)
[![github](https://img.shields.io/github/stars/LqNoob/Neural-Codec-and-Speech-Language-Models.svg?style=social)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models)
[![GitHub issues](https://img.shields.io/github/issues/LqNoob/Neural-Codec-and-Speech-Language-Models?color=critical&label=Issues)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/LqNoob/Neural-Codec-and-Speech-Language-Models?color=success&label=Issues)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub contributors](https://img.shields.io/github/contributors/LqNoob/Neural-Codec-and-Speech-Language-Models.svg)](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models/graphs/contributors/)
</h5>

- [Awesome Codec, TTS & Speech LM](awesome-codec-,-tts-&-speech-lm)
  - [Neural Codec Models](#neural-codec-models)
  - [Zero-Shot Text-to-Speech Synthesizers](#zero-shot-text-to-speech-synthesizers)
  - [Speech Language Models](#speech-language-models)
    - [End-to-End (Paradigms) Models](#end-to-end-paradigms-models)
    - [Cascaded (Paradigms) Models](#cascaded-paradigms-models)
  - [Benchmark](#benchmark)
  - [Survey](#survey)
- [Music Generation](#music-generation)
- [Some Interesting Models](#some-interesting-models)
- [Speech DataSet](speech-dataset)
- [Some Interesting knowledge](some-interesting-knowledge)
  - [Blog & Courses](blog-&-courses)
  - [Minor Points of Concern](#minor-points-of-concern)
- [Reference](#reference)



## Awesome Codec, TTS & Speech LM

1. **Acoustic Tokens**: Acoustic tokens focuses on speech compression and reconstruction, which rely on encoder-decoder architectures with residual vector quantization (RVQ). Specifically, these models quantify speech features (which are downsampled from raw wavforms by one encoder) into a series of discrete tokens, then use one decoder to upsample these discrete tokens into the speech, calculating the reconstruction loss against the original signal. By this approach, we can get discrete acoustic tokens with impressive compression rates and high-fidelity acoustic information, making it more suitable for tasks such as speech synthesis and emotion analysis. (*requires maintaining reconstruction ability and a low bitrate*)
2. **Semantic Tokens**: Semantic tokens involve applying clustering algorithms such as K-means to extract features from self-supervised learning models, using the cluster indices as discrete representations. And it is prediction-based modeling, these models are trained for representation learning by predicting future frames in an autoregressive manner or by using surrounding frames to predict masked frames. This approach tends to prioritize capturing linguistic information within speech, making it particularly useful for recognition and understanding tasks.
4. **Speech Large Language Models**: These models are trained on top of speech and acoustic tokens in a language modeling approach. They demonstrate proficiency in tasks on speech understanding and speech generation. (From [speech-trident](https://github.com/ga642381/speech-trident))

### Neural Codec Models

- [2025/04] **DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation** [[paper](https://openreview.net/forum?id=P7VkjAVClZ)][[demo](https://dualcodec.github.io/)] *Code Comming Soon*
- [2025/04] **ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling** [[paper](https://arxiv.org/abs/2504.10344)][[demo](https://dongchaoyang.top/ALMTokenizer/)]
- [2025/04] **A Streamable Neural Audio Codec with Residual Scalar-Vector Quantization for Real-Time Communication** [[paper](https://arxiv.org/abs/2504.06561)][[demo](https://pb20000090.github.io/StreamCodec/)]
- [2025/04] **One Quantizer is Enough: Toward a Lightweight Audio Codec** [[paper](https://arxiv.org/abs/2504.04949)][[code](https://github.com/zhai-lw/SQCodec)] *SQCodec* :heavy_check_mark:
- [2025/03] **QINCODEC: Neural Audio Compression with Implicit Neural Codebooks** [[paper](https://arxiv.org/abs/2503.19597)][[demo](https://zinebl-sony.github.io/post-training-rvq/)]
- [2025/03] **STFTCodec: High-Fidelity Audio Compression through Time-Frequency Domain Representation** [[paper](https://www.arxiv.org/abs/2503.16989)] 
- [2025/03] **Designing Neural Synthesizers for Low Latency Interaction** [[paper](https://www.arxiv.org/abs/2503.11562)][[code](https://github.com/fcaspe/BRAVE)][[demo](https://fcaspe.github.io/brave/)] :heavy_check_mark:
- [2025/03] **Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens** [[paper](https://arxiv.org/abs/2503.01710)][[code](https://github.com/SparkAudio/Spark-TTS)][[demo](https://sparkaudio.github.io/spark-tts/)] *BiCodec* :heavy_check_mark:
- [2025/03] **FlowDec: A flow-based full-band general audio codec with high perceptual quality** [[paper](https://arxiv.org/abs/2503.01485)][[code](https://github.com/facebookresearch/FlowDec)][[demo](https://sp-uhh.github.io/FlowDec/)] :heavy_check_mark:
- [2025/03] **UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation** [[paper](https://arxiv.org/abs/2503.00733)][[demo](https://alexander-h-liu.github.io/uniwav-demo.github.io/)]
- [2025/02] **UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook** [[paper](https://arxiv.org/abs/2502.20067)][[code](https://github.com/Jiang-Yidi/UniCodec)] *Code Comming Soon*
- [2023/08] **RepCodec: A Speech Representation Codec for Speech Tokenization** [[paper](https://arxiv.org/abs/2309.00169)][[code](https://github.com/mct10/RepCodec)] *semantic speech tokenization* :heavy_check_mark:
- [2024/04] **SRCodec: Split-Residual Vector Quantization for Neural Speech Codec** [[paper](https://ieeexplore.ieee.org/document/10445966)][[demo](https://exercise-book-yq.github.io/SRCodec-demo/)]
- [2024/09] **Low Frame-rate Speech Codec: a Codec Designed for Fast High-quality Speech LLM Training and Inference** [[paper](https://arxiv.org/abs/2409.12117)][[code](https://github.com/NVIDIA/NeMo)][[demo](https://edresson.github.io/Low-Frame-rate-Speech-Codec/)] :heavy_check_mark:
- [2025/02] **FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks** [[paper](https://arxiv.org/abs/2502.04465)][[code](https://github.com/lucadellalib/focalcodec)][[demo](https://lucadellalib.github.io/focalcodec-web/)] :heavy_check_mark:
- [2025/02] **Vector Quantization by Distribution Matching** [[paper](https://openreview.net/forum?id=nS2DBNydCC)][[code](https://github.com/Everlyn-Labs/Wasserstein-VQ)] *Wasserstein-VQ* | *codebook collapse* :heavy_check_mark:
- [2024/12] **Preventing Local Pitfalls in Vector Quantization via Optimal Transport** [[paper](https://arxiv.org/abs/2412.15195)][[code](https://github.com/zbr17/OptVQ)] *OptVQ* | *codebook collapse* :heavy_check_mark:
- [2025/02] **ComplexDec: A Domain-robust High-fidelity Neural Audio Codec with Complex Spectrum Modeling** [[paper](https://www.arxiv.org/abs/2502.02019)][[demo](https://bigpon.github.io/ComplexDec_demo/)]
- [2025/02] **Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis** [[paper](https://arxiv.org/abs/2502.04128)][[code](https://github.com/zhenye234/X-Codec-2.0)][[infer-unofficial](https://github.com/nivibilla/local-llasa-tts)] *X-Codec-2.0* :heavy_check_mark:
- [2025/01] **MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization** [[paper](https://arxiv.org/abs/2501.01108)][[code](https://github.com/tencent-ailab/MuQ)] :heavy_check_mark:
- [2025/01] **SECodec: Structural Entropy-based Compressive Speech Representation Codec for Speech Language Models** [[paper](https://arxiv.org/abs/2501.00018)][[code](https://github.com/wlq2019/SECodec)] *`Code Comming Soon`*
- [2024/12] **FreeCodec: A disentangled neural speech codec with fewer tokens** [[paper](https://arxiv.org/abs/2412.01053)][[code](https://github.com/exercise-book-yq/FreeCodec)][[demo](https://exercise-book-yq.github.io/FreeCodec-Demo/)] `Code Comming Soon` | *speaker encoder, content encoder and prosody encoder*
- [2024/11] **TS3-Codec: Transformer-Based Simple Streaming Single Codec** [[paper](https://arxiv.org/abs/2411.18803)] *free-convolution*
- [2024/11] **Scaling Transformer for Low-bitrate High-Quality Speech Coding** [[paper](https://arxiv.org/abs/2411.19842)][[code](https://github.com/Stability-AI/stable-codec)][[demo](https://stability-ai.github.io/stable-codec-demo/)] *transformer-based and scale it into 1B parameter range* :heavy_check_mark:
- [2024/11] **PyramidCodec: Hierarchical Codec for Long-form Music Generation in Audio Domain** [[paper](https://aclanthology.org/2024.findings-emnlp.246/)][[demo](https://pyramidcodec.github.io/)] `Code Comming Soon` | *Music Tokenizer, Similar to MsCodec*
- [2024/11] **Wavehax: Aliasing-Free Neural Waveform Synthesis Based on 2D Convolution and Harmonic Prior for Reliable Complex Spectrogram Estimation** [[paper](https://arxiv.org/abs/2411.06807)][[code](https://github.com/chomeyama/wavehax)][[demo](https://chomeyama.github.io/wavehax-demo/)] *aliasing-free* :heavy_check_mark:
- [2024/11] **VChangeCodec: A High-efficiency Neural Speech Codec with Built-in Voice Changer for Real-time Communication** [[paper](https://openreview.net/forum?id=qDSfOQBrOD)][[demo](https://anonymous666-speech.github.io/Demo-VChangeCodec/)] *integrates the Voice Changer model directly into the speech Codec*
- [2024/11] **Towards Codec-LM Co-design for Neural Codec Language Models** [[paper](https://openreview.net/forum?id=KCVv3tICvp)] `Code Comming Soon` | *proposing several codec-LM co-design strategies*
- [2024/11] **Universal Speech Token Learning via Low-Bitrate Neural Codec and Pretrained Representations** [[paper](https://arxiv.org/abs/2503.12115)][[IEEE](https://ieeexplore.ieee.org/abstract/document/10738376?casa_token=eWtmSXEr4AEAAAAA:FzYuQIESJ2LXwl9smJQe3RakpDUFuJ-AS0d39ZDlhsI0tBVX_8P7hu4a59yZezz7hpYd3VomUDo)] *UniCodec* ｜ *several information-disentangled discrete tokens, similar to ns3_codec*
- [2024/11] **hertz-dev** [[code](https://github.com/Standard-Intelligence/hertz-dev)] *WaveCodec* :heavy_check_mark:
- [2024/11] **SimVQ: Addressing Representation Collapse in Vector Quantized Models with One Linear Layer** [[paper](https://arxiv.org/abs/2411.02038)][[code](https://github.com/youngsheen/SimVQ)] *codebook collapse* :heavy_check_mark:
- [2024/11] **MDCTCodec: A Lightweight MDCT-based Neural Audio Codec towards High Sampling Rate and Low Bitrate Scenarios** [[paper](https://arxiv.org/abs/2411.00464)][[demo](https://pb20000090.github.io/MDCTCodecSLT2024/)] *discrete cosine transform (MDCT) as input*
- [2024/10] **Accelerating Codec-based Speech Synthesis with Multi-Token Prediction and Speculative Decoding** [[paper](https://arxiv.org/abs/2410.13839)][[demo](https://multpletokensprediction.github.io/multipletokensprediction.github.io/)] *`Code Comming Soon`*
- [2024/10] **Pushing the frontiers of audio generation** [[blog](https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/)] *google deepmind*
- [2024/11] **DC-Spin: A Speaker-invariant Speech Tokenizer for Spoken Language Models** [[paper](https://arxiv.org/abs/2410.24177)] *Double-Codebook Speaker-invariant Clustering*
- [2024/10] **A Closer Look at Neural Codec Resynthesis: Bridging the Gap between Codec and Waveform Generation** [[paper](https://arxiv.org/abs/2410.22448)][[demo](https://alexander-h-liu.github.io/codec-resyn.github.io/)] *Is predicting the remaining RVQ codes necessary?*
- [2024/10] **APCodec+: A Spectrum-Coding-Based High-Fidelity and High-Compression-Rate Neural Audio Codec with Staged Training Paradigm** [[paper](https://arxiv.org/abs/2410.22807)][[demo](https://redmist328.github.io/APCodecPlus-demo/)] *two-stage joint-individual training paradigm*
- [2024/10] **Optimizing Neural Speech Codec for Low-Bitrate Compression via Multi-Scale Encoding** [[paper](https://arxiv.org/abs/2410.15749)][[demo](https://tencentgamemate.github.io/MsCodec-Demo/)] *MsCodec, Multi-Scale Encoding*
- [2024/10] **LSCodec: Low-Bandwidth and Speaker-Decoupled Discrete Speech Codec** [[paper](https://arxiv.org/abs/2410.15764)][[demo](https://cantabile-kwok.github.io/LSCodec/)] *speaker timbre decouple*
- [2024/10] **DM-Codec: Distilling Multimodal Representations for Speech Tokenization** [[paper](https://arxiv.org/abs/2410.15017)][[code](https://github.com/mubtasimahasan/DM-Codec)] *acoustic properties, semantic meaning, and contextual clues* :heavy_check_mark:
- [2024/10] **ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization for Neural Audio Codecs** [[paper](https://arxiv.org/abs/2410.12359)][[demo](https://anonymous.4open.science/w/ERVQ-A907/)] *address codebook collapse based on intra- and inter-codebook optimization*
- [2024/10] **Code Drift: Towards Idempotent Neural Audio Codecs** [[paper](https://arxiv.org/abs/2410.11025)][[demo](https://oreillyp.github.io/codedrift/)] *Idempotence – the stability of a codec’s decoded output under multiple rounds of encoding and decoding*
- [2021/10] **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** [[paper](https://arxiv.org/abs/2110.13900)][[code](https://github.com/microsoft/unilm/tree/master/wavlm)] *semantic information & content generation* :heavy_check_mark:
- [2021/08] **W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training** [[paper](https://arxiv.org/abs/2108.06209)]
- [2021/06] **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** [[paper](https://arxiv.org/abs/2106.07447)][[code](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)] *semantic information & content generation* :heavy_check_mark:
- [2020/06] **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** [[paper](https://arxiv.org/abs/2006.11477)][[code](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)] :heavy_check_mark:
- [2024/10] **Low Bitrate High-Quality RVQGAN-based Discrete Speech Tokenizer** [[paper](https://arxiv.org/abs/2410.08325)][[code](https://huggingface.co/ibm/DAC.speech.v1.0)][[demo](https://s3.us-south.objectstorage.softlayer.net/zk-wav-data/Webpages/SpeechDAC_IS2024/index.html)] *finetuned-version of DAC* :heavy_check_mark:
- [2024/09] **BigCodec: Pushing the Limits of Low-Bitrate Neural Speech Codec** [[paper](https://arxiv.org/abs/2409.05377)][[code](https://github.com/Aria-K-Alethia/BigCodec)][[demo](https://aria-k-alethia.github.io/bigcodec-demo/)]  *low-bitrate neural speech codec*  :heavy_check_mark:
- [2024/10] **Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models** [[paper](https://arxiv.org/abs/2409.19283)][[demo](https://consistencyinneuralcodec.github.io/)] *Inconsistency*
- [2024/09] **Reverse Engineering of Supervised Semantic Speech Tokenizer (S3Tokenizer) proposed in CosyVoice** [[code](https://github.com/xingchensong/S3Tokenizer)] *S3Tokenizer* :heavy_check_mark:
- [2024/09] **FlowMAC: Conditional Flow Matching for Audio Coding at Low Bit Rates** [[paper](https://arxiv.org/abs/2409.17635)]  *Flow Matching*
- [2024/09] **ESPnet-Codec: Comprehensive Training and Evaluation of Neural Codecs for Audio, Music, and Speech** [[paper](https://arxiv.org/abs/2409.15897)][[code](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/codec1)] *Comprehensive Platform* :heavy_check_mark:
- [2024/09] **MuCodec: Ultra Low-Bitrate Music Codec** [[paper](https://arxiv.org/abs/2409.13216)][[code](https://github.com/xuyaoxun/MuCodec)][[demo](https://xuyaoxun.github.io/MuCodec_demo/)] *Music Codec* :heavy_check_mark:
- [2024/09] **Audio Codec Augmentation for Robust Collaborative Watermarking of Speech Synthesis** [[paper](https://arxiv.org/abs/2409.13382)][[code](https://github.com/ljuvela/collaborative-watermarking-with-codecs)][[demo](https://ljuvela.github.io/collaborative-watermarking-with-codecs-demo/)] *Watermarking* :heavy_check_mark:
- [2024/09] **NDVQ: Robust Neural Audio Codec with Normal Distribution-Based Vector Quantization** [[paper](https://arxiv.org/abs/2409.12717)][[code](https://github.com/ZhikangNiu/NDVQ)] `Code Comming Soon`
- [2024/09] **Speaking from Coarse to Fine: Improving Neural Codec Language Model via Multi-Scale Speech Coding and Generation** [[paper](https://arxiv.org/abs/2409.11630v1)][[demo](https://hhguo.github.io/DemoCoFiSpeech/)] *CoFi-Speech*
- [2024/09] **SoCodec: A Semantic-Ordered Multi-Stream Speech Codec for Efficient Language Model Based Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2409.00933)][[code](https://github.com/hhguo/SoCodec)][[demo](https://hhguo.github.io/DemoSoCodec/)] :heavy_check_mark:
- [2024/08] **Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model** [[paper](https://arxiv.org/abs/2408.17175)][[code](https://github.com/zhenye234/xcodec)][[demo](https://x-codec-audio.github.io/)] *X-Codec* :heavy_check_mark:
- [2024/08] **WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling** [[paper](https://arxiv.org/abs/2408.16532)][[code](https://github.com/jishengpeng/WavTokenizer)][[demo](https://wavtokenizer.github.io/)] :heavy_check_mark:
- [2024/08] **Music2Latent: Consistency Autoencoders for Latent Audio Compression** [[paper](https://www.arxiv.org/abs/2408.06500)][[code](https://github.com/SonyCSLParis/music2latent)][[demo](https://sonycslparis.github.io/music2latent-companion/)] *continuous latent space* :heavy_check_mark:
- [2024/08] **SimpleSpeech 2: Towards Simple and Efficient Text-to-Speech with Flow-based Scalar Latent Transformer Diffusion Models** [[paper](https://arxiv.org/abs/2408.13893)][[demo](https://dongchaoyang.top/SimpleSpeech2_demo/)]
- [2024/06] **SimpleSpeech: Towards Simple and Efficient Text-to-Speech with Scalar Latent Transformer Diffusion Models** [[paper](https://arxiv.org/abs/2406.02328v2)][[code](https://github.com/yangdongchao/SimpleSpeech)][[demo](https://simplespeech.github.io/simplespeechDemo/)] *SQ-Codec* | `Code Comming Soon`
- [2024/02] **Language-Codec: Reducing the Gaps Between Discrete Codec Representation and Speech Language Models** [[paper](https://arxiv.org/abs/2402.12208)][[code](https://github.com/jishengpeng/Languagecodec)][[demo](https://languagecodec.github.io/)] :heavy_check_mark:
- [2024/04] **ESC: Efficient Speech Coding with Cross-Scale Residual Vector Quantized Transformers** [[paper](https://arxiv.org/abs/2404.19441)][[code](https://github.com/yzGuu830/efficient-speech-codec)] :heavy_check_mark:
- [2024/07] **SuperCodec: A Neural Speech Codec with Selective Back-Projection Network** [[paper](https://arxiv.org/abs/2407.20530)][[code](https://github.com/exercise-book-yq/Supercodec)][[demo](https://exercise-book-yq.github.io/SuperCodec-Demo/)] :heavy_check_mark:
- [2024/07] **dMel: Speech Tokenization made Simple** [[paper](https://arxiv.org/abs/2407.15835)] `Code Comming Soon`
- [2024/02] **APCodec: A Neural Audio Codec with Parallel Amplitude and Phase Spectrum Encoding and Decoding** [[paper](https://arxiv.org/abs/2402.10533)][[code](https://github.com/YangAi520/APCodec)][[demo](https://yangai520.github.io/APCodec/)] :heavy_check_mark:
- [2024/06] **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** [[paper](https://www.arxiv.org/abs/2406.07422)][[demo](https://kkksuper.github.io/Single-Codec/)]
- [2024/07] **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** [[paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf)][[code](https://github.com/FunAudioLLM/CosyVoice)][[demo](https://fun-audio-llm.github.io/)] :heavy_check_mark:
- [2023/06] **Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis** [[paper](https://arxiv.org/abs/2306.00814)][[code](https://github.com/gemelo-ai/vocos)][[demo](https://gemelo-ai.github.io/vocos/)] :heavy_check_mark:
- [2024/04] **SNAC: Multi-Scale Neural Audio Codec** [[paper](https://www.arxiv.org/abs/2410.14411)][[code](https://github.com/hubertsiuzdak/snac)][[demo](https://hubertsiuzdak.github.io/snac/)] :heavy_check_mark:
- [2024/06] **UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner** [[paper](https://arxiv.org/abs/2406.10056)][[code](https://github.com/yangdongchao/LLM-Codec)] *LLM-Codec* :heavy_check_mark:
- [2024/01] **Finite Scalar Quantization: VQ-VAE Made Simple** [[paper](https://openreview.net/forum?id=8ishA3LxN8)][[code](https://github.com/google-research/google-research/tree/master/fsq)] *FSQ, no codebook collapse* :heavy_check_mark:
- [2024/06] **Spectral Codecs: Spectrogram-Based Audio Codecs for High Quality Speech Synthesis** [[paper](https://arxiv.org/abs/2406.05298)][[code](https://github.com/NVIDIA/NeMo)][[demo](https://rlangman.github.io/spectral-codec/)] :heavy_check_mark:
- [2023/09] **Generative Pre-trained Speech Language Model with Efficient Hierarchical Transformer** [[paper](https://openreview.net/forum?id=TJNCnkDRkY)]
- [2024/06] **BiVocoder: A Bidirectional Neural Vocoder Integrating Feature Extraction and Waveform Generation** [[paper](https://arxiv.org/abs/2406.02162)][[demo](https://redmist328.github.io/BiVcoder_demo)]
- [2024/04] **The X-LANCE Technical Report for Interspeech 2024 Speech Processing Using Discrete Speech Unit Challenge** [[paper](https://arxiv.org/abs/2404.06079v2)]
- [2023/06] **UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding** [[paper](https://arxiv.org/abs/2306.07547v6)][[code](https://github.com/X-LANCE/UniCATS-CTX-vec2wav)][[demo](https://cpdu.github.io/unicats/)] *acoustic model CTX-txt2vec and vocoder CTX-vec2wav | speech continuation and editing | similar to Encoder-Decoder* :heavy_check_mark:
- [2024/06] **Addressing Index Collapse of Large-Codebook Speech Tokenizer with Dual-Decoding Product-Quantized Variational Auto-Encoder** [[paper](https://arxiv.org/abs/2406.02940)]
- [2024/06] **Coding Speech through Vocal Tract Kinematics** [[paper](https://arxiv.org/abs/2406.12998)][[code](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding)] :heavy_check_mark:
- [2024/05] **HILCodec: High Fidelity and Lightweight Neural Audio Codec** [[paper](https://arxiv.org/abs/2405.04752)][[code](https://github.com/aask1357/hilcodec)][[demo](https://aask1357.github.io/hilcodec/)] :heavy_check_mark:
- [2024/04] **SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound** [[paper](https://arxiv.org/abs/2405.00233)][[code](https://github.com/haoheliu/SemantiCodec)][[demo](https://haoheliu.github.io/SemantiCodec/)] :heavy_check_mark:
- [2024/01] **Residual Quantization with Implicit Neural Codebooks** [[paper](https://arxiv.org/abs/2401.14732)][[code](https://github.com/facebookresearch/Qinco)] *Qinco* :heavy_check_mark:
- [2024/01] **SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models** [[paper](https://openreview.net/forum?id=AF9Q8Vip84)][[code](https://github.com/ZhangXInFD/SpeechTokenizer)][[demo](https://0nutation.github.io/SpeechTokenizer.github.io/)] :heavy_check_mark:
- [2024/01] **Residual Quantization with Implicit Neural Codebooks** [[paper](https://arxiv.org/abs/2401.14732)][[code](https://github.com/facebookresearch/Qinco)] :heavy_check_mark:
- [2023/10] **Acoustic BPE for Speech Generation with Discrete Tokens** [[paper](https://arxiv.org/abs/2310.14580)][[code](https://github.com/AbrahamSanders/codec-bpe)] :heavy_check_mark:
- [2023/09] **BANC: Towards Efficient Binaural Audio Neural Codec for Overlapping Speech** [[paper](https://arxiv.org/abs/2309.07416)][[code](https://github.com/anton-jeran/MULTI-AUDIODEC)][[demo](https://anton-jeran.github.io/MAD/)] :heavy_check_mark:
- [2023/09] **Fewer-token Neural Speech Codec with Time-invariant Codes** [[paper](https://arxiv.org/abs/2310.00014)][[code](https://github.com/y-ren16/TiCodec)][[demo](https://y-ren16.github.io/TiCodec/)] *Ti-Codec* :heavy_check_mark:
- [2023/09] **FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec** [[paper](https://arxiv.org/abs/2309.07405v2)][[code](https://github.com/modelscope/FunCodec)][[demo](https://funcodec.github.io/)] :heavy_check_mark:
- [2023/09] **High Fidelity Neural Audio Compression** [[paper](https://openreview.net/forum?id=ivCd8z8zR2)][[code](https://github.com/facebookresearch/encodec)][[code-Unofficial](https://github.com/ZhikangNiu/encodec-pytorch)] [[demo](https://ai.honu.io/papers/encodec/samples.html)] *Encodec* :heavy_check_mark:
- [2023/09] **Soundstorm: Efficient parallel audio generation** [[paper](https://openreview.net/forum?id=KknWbD5j95)][[demo](https://google-research.github.io/seanet/soundstorm/examples/)]
- [2023/09] **High-Fidelity Audio Compression with Improved RVQGAN** [[paper](https://openreview.net/forum?id=qjnl1QUnFA)][[code](https://github.com/descriptinc/descript-audio-codec)][[demo](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5)] *DAC* :heavy_check_mark:
- [2023/09] **SpatialCodec: Neural Spatial Speech Coding** [[paper](https://arxiv.org/abs/2309.07432)][[code](https://github.com/XZWY/SpatialCodec)][[demo](https://xzwy.github.io/SpatialCodecDemo/)] :heavy_check_mark:
- [2023/05] **HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec** [[paper](https://arxiv.org/abs/2305.02765v2)][[code](https://github.com/yangdongchao/AcademiCodec)] *AcademiCodec & Group-RVQ* :heavy_check_mark:
- [2023/05] **AudioDec: An Open-source Streaming High-fidelity Neural Audio Codec** [[paper](https://arxiv.org/abs/2305.16608)][[code](https://github.com/facebookresearch/AudioDec)][[demo](https://bigpon.github.io/AudioDec_demo/)] :heavy_check_mark:
- [2023/01] **InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt** [[paper](https://arxiv.org/abs/2301.13662v2)][[code](https://github.com/yangdongchao/InstructTTS)][[demo](https://dongchaoyang.top/InstructTTS/)] :heavy_check_mark:
- [2022/09] **AudioLM: a Language Modeling Approach to Audio Generation** [[paper](https://arxiv.org/abs/2209.03143v2)][[demo](https://google-research.github.io/seanet/audiolm/examples/)]
- [2021/07] **SoundStream: An End-to-End Neural Audio Codec** [[paper](https://arxiv.org/abs/2107.03312)][[code](https://github.com/google/lyra)][[demo](https://google-research.github.io/seanet/soundstream/examples/)] :heavy_check_mark:


### Zero-Shot Text-to-Speech Synthesizers

- [2025/04] **SLED-TTS: Efficient Speech Language Modeling via Energy Distance in Continuous Space** [[code](https://github.com/ictnlp/SLED-TTS)][[demo](https://sled-demo.github.io/)]
- [2025/04] **dia** [[code](https://github.com/nari-labs/dia)][[demo](https://yummy-fir-7a4.notion.site/dia)]
- [2025/04] **EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting** [[paper](https://arxiv.org/abs/2504.12867)][[demo](https://anonymous.4open.science/w/EmoVoice-DF55/)] *Code Comming Soon*
- [2025/04] **Pseudo-Autoregressive Neural Codec Language Models for Efficient Zero-Shot Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2504.10352)][[demo](https://anonymous-palle.github.io/)]
- [2025/04] **AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2504.10309)][[demo](https://thuhcsi.github.io/icme2025-AutoStyle-TTS/)]
- [2025/04] **RWKVTTS: Yet another TTS based on RWKV-7** [[paper](https://arxiv.org/abs/2504.03289)][[code](https://github.com/yynil/RWKVTTS)] :heavy_check_mark:
- [2025/04] **F5R-TTS: Improving Flow Matching based Text-to-Speech with Group Relative Policy Optimization** [[paper](https://arxiv.org/abs/2504.02407)][[demo](https://frontierlabs.github.io/F5R/)]
- [2025/04] **MegaTTS 3: Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis** [[paper](https://arxiv.org/abs/2502.18924)][[code](https://github.com/bytedance/MegaTTS3)][[demo](https://sditdemo.github.io/sditdemo/)] :heavy_check_mark:
- [2025/03] **FireRedTTS-1S: An Upgraded Streamable Foundation Text-to-Speech System** [[paper](https://arxiv.org/abs/2503.20499)][[demo](https://fireredteam.github.io/demos/firered_tts_1s/)]
- [2025/03] **MoonCast: High-Quality Zero-Shot Podcast Generation** [[paper](https://arxiv.org/abs/2503.14345)][[code](https://github.com/jzq2000/MoonCast)][[demo](https://mooncastdemo.github.io/)] :heavy_check_mark:
- [2025/03] **Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens** [[paper](https://arxiv.org/abs/2503.01710)][[code](https://github.com/SparkAudio/Spark-TTS)][[demo](https://sparkaudio.github.io/spark-tts/)] :heavy_check_mark:
- [2025/02] **Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis** [[paper](https://www.arxiv.org/abs/2502.18924)][[demo](https://sditdemo.github.io/sditdemo/)]
- [2025/02] **SyncSpeech: Low-Latency and Efficient Dual-Stream Text-to-Speech based on Temporal Masked Transformer** [[paper](https://arxiv.org/abs/2502.11094)][[demo](https://syncspeech.github.io/)] *Code Comming Soon*
- [2025/02] **FELLE: Autoregressive Speech Synthesis with Token-Wise Coarse-to-Fine Flow Matching** [[paper](https://arxiv.org/abs/2502.11128)][[demo](https://felle-demo.github.io/)]
- [2025/02] **IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System** [[paper](https://arxiv.org/abs/2502.05512)][[code](https://github.com/index-tts/index-tts)][[demo](https://index-tts.github.io/)] :heavy_check_mark:
- [2025/02] **Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance** [[paper](https://arxiv.org/abs/2502.05236)][[demo](https://koeltts.github.io/)] *Code Comming Soon*
- [2025/02] **Beta Release of Zonos-v0.1** [[blog](https://www.zyphra.com/post/beta-release-of-zonos-v0-1)][[code](https://github.com/Zyphra/Zonos)] :heavy_check_mark:
- [2025/01] **Kokoro-TTS** [[checkpoints](https://huggingface.co/hexgrad/Kokoro-82M)][[huggingface](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)]
- [2025/01] **MARS6: A Small and Robust Hierarchical-Codec Text-to-Speech Model** [[paper](https://arxiv.org/abs/2501.05787)][[demo](https://camb-ai.github.io/mars6-turbo/)] *`Code Comming Soon`*
- [2025/01] **DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions** [[paper](https://arxiv.org/abs/2501.04256)][[code](https://github.com/HappyColor/DrawSpeech_PyTorch)][[demo](https://happycolor.github.io/DrawSpeech/)] :heavy_check_mark:
- [2025/01] **TouchTTS: An Embarrassingly Simple TTS Framework that Everyone Can Touch** [[paper](https://www.arxiv.org/abs/2412.08237)] *data scaling and deployment efficiency*
- [2024/12] **CrossSpeech++: Cross-lingual Speech Synthesis with Decoupled Language and Speaker Generation** [[paper](https://arxiv.org/abs/2412.20048)][[demo](https://mm.kaist.ac.kr/projects/CrossSpeechpp/)]
- [2024/12] **VoiceDiT: Dual-Condition Diffusion Transformer for Environment-Aware Speech Synthesis** [[paper](https://arxiv.org/abs/2412.19259)][[demo](https://mm.kaist.ac.kr/projects/voicedit/)] *environment-aware speech synthesis*
- [2024/12] **Autoregressive Speech Synthesis with Next-Distribution Prediction** [[paper](https://www.arxiv.org/abs/2412.16846)][[demo](https://zxf-icpc.github.io/kalle/)] *KALL-E*
- [2024/12] **ProsodyFM: Unsupervised Phrasing and Intonation Control for Intelligible Speech Synthesis** [[paper](https://www.arxiv.org/abs/2412.11795)][[code](https://github.com/XianghengHee/ProsodyFM)][[demo](https://sordid-eggplant-f05.notion.site/Demo-for-ProsodyFM-6140dee6fe4a4e5eab1395867d7570f8)] *Code Comming Soon*
- [2024/12] **Interleaved Speech-Text Language Models are Simple Streaming Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2412.16102)] *IST-LM*
- [2024/12] **CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models** [[paper](https://arxiv.org/abs/2412.10117)][[code](https://github.com/FunAudioLLM/CosyVoice)][[demo](https://funaudiollm.github.io/cosyvoice2/)] :heavy_check_mark:
- [2024/12] **TouchTTS: An Embarrassingly Simple TTS Framework that Everyone Can Touch** [[paper](https://www.arxiv.org/abs/2412.08237)]
- [2024/11] **Visatronic: A Multimodal Decoder-Only Model for Speech Synthesis** [[paper](https://www.arxiv.org/abs/2411.17690)] *Code Comming Soon* | *Text & Video to Speech*
- [2024/11] **Debatts: Zero-Shot Debating Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2411.06540)][[demo](https://amphionspace.github.io/debatts/#demos)] *Debating TTS & Dataset*
- [2024/11] **OuteTTS-0.1-350M** [[blog](https://www.outeai.com/blog/OuteTTS-0.1-350M)][[code](https://github.com/edwko/OuteTTS)] :heavy_check_mark:
- [2024/12] **The Codec Language Model-based Zero-Shot Spontaneous Style TTS System for CoVoC Challenge 2024** [[paper](https://www.arxiv.org/abs/2412.01100)] *ISCSLP 2024*
- [2024/10] **The ISCSLP 2024 Conversational Voice Clone (CoVoC) Challenge: Tasks, Results and Findings** [[paper](https://arxiv.org/abs/2411.00064)] *zero-shot spontaneous style voice cloning* | *ISCSLP 2024*
- [2024/07] **ICAGC 2024: Inspirational and Convincing Audio Generation Challenge 2024** [[paper](https://arxiv.org/abs/2407.12038)] *emotional & background audio generation* | *ISCSLP 2024*
- [2024/11] **Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2411.01156)][[code](https://github.com/fishaudio/fish-speech)] :heavy_check_mark:
- [2024/10] **STTATTS: Unified Speech-To-Text And Text-To-Speech Model** [[paper](https://arxiv.org/abs/2410.18607)][[code](https://github.com/mbzuai-nlp/sttatts)]
- [2024/10] **SPIRIT LM: Interleaved Spoken and Written Language Model** [[paper](https://arxiv.org/abs/2402.05755)][[code](https://github.com/facebookresearch/spiritlm)][[demo](https://speechbot.github.io/spiritlm/)] :heavy_check_mark:
- [2023/05] **Better speech synthesis through scaling** [[paper](https://arxiv.org/abs/2305.07243)][[code](https://github.com/neonbjb/tortoise-tts)][[blog](https://nonint.com/2022/06/18/lab-notes-cheater-latents/)] *Tortoise TTS* :heavy_check_mark:
- [2024/10] **F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching** [[paper](https://arxiv.org/abs/2410.06885)][[code](https://github.com/SWivid/F5-TTS)][[demo](https://swivid.github.io/F5-TTS/)] :heavy_check_mark:
- [2024/09] **Takin: A Cohort of Superior Quality Zero-shot Speech Generation Models** [[paper](https://arxiv.org/abs/2409.12139)][[demo](https://takinaudiollm.github.io/)]
- [2024/09] **FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications** [[paper](https://www.arxiv.org/abs/2409.03283)][[code](https://github.com/FireRedTeam/FireRedTTS)][[demo](https://fireredteam.github.io/demos/firered_tts/)] *voice cloning for dubbing and human-like speech generation for chatbots* :heavy_check_mark:
- [2024/09] **MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer** [[paper](https://arxiv.org/abs/2409.00750)][[code](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)][[demo](https://maskgct.github.io/)] *Masked Generative Model* | *Similar to Seed-TTS* :heavy_check_mark:
- [2024/08] **VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling** [[paper](https://www.arxiv.org/abs/2408.15676)][[code](https://github.com/thuhcsi/VoxInstruct)][[demo](https://voxinstruct.github.io/VoxInstruct/)] :heavy_check_mark:
- [2024/08] **Bailing-TTS: Chinese Dialectal Speech Synthesis Towards Human-like Spontaneous Representation** [[paper](https://arxiv.org/abs/2408.00284)][[demo](https://c9412600.github.io/bltts_tech_report/index.html)]
- [2024/04] **FlashSpeech: Efficient Zero-Shot Speech Synthesis** [[paper](https://arxiv.org/abs/2404.14700)][[code](https://github.com/zhenye234/FlashSpeech)][[demo](https://flashspeech.github.io/)] :heavy_check_mark:
- [2024/07] **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** [[paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf)] [[code](https://github.com/FunAudioLLM/CosyVoice)][[demo](https://funaudiollm.github.io/)] :heavy_check_mark:
- [2024/07] **Robust Zero-Shot Text-to-Speech Synthesis with Reverse Inference Optimization** [[paper](https://arxiv.org/abs/2407.02243)][[demo](https://yuchen005.github.io/RIO-TTS-demos/)] *Human FeedBack*
- [2024/06] **E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS** [[paper](https://arxiv.org/abs/2406.18009)][[demo](https://www.microsoft.com/en-us/research/project/e2-tts/)] *similar to Seed-TTS*
- [2023/11] **HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis** [[paper](https://arxiv.org/abs/2311.12454)][[code](https://github.com/sh-lee-prml/HierSpeechpp)][[demo](https://sh-lee-prml.github.io/HierSpeechpp-demo/)] :heavy_check_mark:
- [2024/06] **TacoLM: GaTed Attention Equipped Codec Language Model are Efficient Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2406.15752)][[code](https://github.com/Ereboas/TacoLM)][[demo](https://ereboas.github.io/TacoLM/)] :heavy_check_mark:
- [2024/01] **CLaM-TTS: Improving Neural Codec Language Model for Zero-Shot Text-to-Speech** [[paper](https://openreview.net/forum?id=ofzeypWosV)][[demo](https://clam-tts.github.io/)]
- [2024/06] **DiTTo-TTS: Efficient and Scalable Zero-Shot Text-to-Speech with Diffusion Transformer** [[paper](https://arxiv.org/abs/2406.11427)][[demo](https://ditto-tts.github.io/)]
- [2024/06] **VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignment** [[paper](https://arxiv.org/abs/2406.07855)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-r/)]
- [2024/06] **Autoregressive Diffusion Transformer for Text-to-Speech Synthesis** [[paper](https://www.arxiv.org/abs/2406.05551)][[demo](https://ardit-tts.github.io/)]
- [2024/06] **VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2406.05370)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/)]
- [2024/06] **XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model** [[paper](https://arxiv.org/abs/2406.04904)][[code](https://github.com/coqui-ai/TTS/tree/main)][[demo](https://edresson.github.io/XTTS/)] :heavy_check_mark:
- [2024/06] **ControlSpeech: Towards Simultaneous Zero-shot Speaker Cloning and Zero-shot Language Style Control With Decoupled Codec** [[paper](https://arxiv.org/abs/2406.01205)][[code](https://github.com/jishengpeng/ControlSpeech)][[demo](https://controlspeech.github.io/)] :heavy_check_mark:
- [2024/08] **SSL-TTS: Leveraging Self-Supervised Embeddings and kNN Retrieval for Zero-Shot Multi-speaker TTS** [[paper](https://www.arxiv.org/abs/2408.10771)][[demo](https://www.arxiv.org/abs/2408.10771)] *SSL*
- [2024/08] **VoiceTailor: Lightweight Plug-In Adapter for Diffusion-Based Personalized Text-to-Speech** [[paper](https://arxiv.org/abs/2408.14739)][[demo](https://voicetailor.github.io/)] *LORA*
- [2024/08] **StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech** [[paper](https://www.arxiv.org/abs/2408.14713)][[demo](https://style-speech.vercel.app/)] *LORA*
- [2024/08] **EELE: Exploring Efficient and Extensible LoRA Integration in Emotional Text-to-Speech** [[paper](https://www.arxiv.org/abs/2408.10852)] *LORA*
- [2024/07] **Spontaneous Style Text-to-Speech Synthesis with Controllable Spontaneous Behaviors Based on Language Models** [[paper](https://arxiv.org/abs/2407.13509)][[demo](https://thuhcsi.github.io/interspeech2024-SponLMTTS/)] *Spontaneous*
- [2024/01] **EmotiVoice 😊: a Multi-Voice and Prompt-Controlled TTS Engine** [[code](https://github.com/netease-youdao/EmotiVoice)] :heavy_check_mark:
- [2024/06] **Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment** [[paper](https://arxiv.org/abs/2406.17957v1)][[demo](https://t5tts.github.io/)] *Monotonic Alignment*
- [2024/01] **Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction** [[paper](https://arxiv.org/abs/2401.01498)][[demo](https://gannnn123.github.io/token-transducer/)] *Transducer/End-to-End*
- [2024/01] **VALL-T: Decoder-Only Generative Transducer for Robust and Decoding-Controllable Text-to-Speech** [[paper](https://arxiv.org/abs/2401.14321)][[code](https://github.com/cpdu/vallt)][[demo](https://cpdu.github.io/vallt/)] *`Code Comming Soon` | Transducer*
- [2024/06] **High Fidelity Text-to-Speech Via Discrete Tokens Using Token Transducer and Group Masked Language Model** [[paper](https://arxiv.org/abs/2406.17310)][[demo](https://srtts.github.io/interpreting-speaking/)] *Transducer/End-to-End*
- [2023/02] **Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision** [[paper](https://arxiv.org/abs/2302.03540)][[code](https://github.com/collabora/WhisperSpeech)][[demo](https://collabora.github.io/WhisperSpeech/)] *SpearTTS | WhisperSpeech* :heavy_check_mark:
- [2024/02] **Natural language guidance of high-fidelity text-to-speech with synthetic annotations** [[paper](https://arxiv.org/abs/2402.01912)][[code](https://github.com/huggingface/parler-tts)][[demo](https://www.text-description-to-speech.com/)] *Prompt Control | Parler-TTS* :heavy_check_mark:
- [2024/06] **WenetSpeech4TTS: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark** [[paper](https://arxiv.org/abs/2406.05763v2)][[demo](https://huggingface.co/Wenetspeech4TTS)] 
- [2024/06] **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** [[paper](https://arxiv.org/abs/2406.02430)][[demo](https://bytedancespeech.github.io/seedtts_tech_report/)]
- [2024/06] **Enhancing Zero-shot Text-to-Speech Synthesis with Human Feedback** [[paper](https://www.arxiv.org/abs/2406.00654)] *Human Feedback*
- [2024/04] **SpeechAlign: Aligning Speech Generation to Human Preferences** [[paper](https://arxiv.org/abs/2404.05600)][[code](https://github.com/0nutation/SpeechGPT)][[demo](https://0nutation.github.io/SpeechAlign.github.io/)] *Human Feedback* :heavy_check_mark:
- [2024/04] **StoryTTS: A Highly Expressive Text-to-Speech Dataset with Rich Textual Expressiveness Annotations** [[paper](https://arxiv.org/abs/2404.14946)][[code](https://github.com/X-LANCE/StoryTTS)][[demo](https://goarsenal.github.io/StoryTTS/)] *Lian Liru(连丽如) dataset* :heavy_check_mark:
- [2024/04] **TextrolSpeech: A Text Style Control Speech Corpus with Codec Language Text-to-Speech Models** [[paper](https://ieeexplore.ieee.org/abstract/document/10445879)][[code](https://github.com/jishengpeng/TextrolSpeech)][[demo](https://sall-e.github.io/)] `Code Comming Soon`
- [2024/03] **HAM-TTS: Hierarchical Acoustic Modeling for Token-Based Zero-Shot Text-to-Speech with Model and Data Scaling** [[paper](https://arxiv.org/abs/2403.05989)][[demo](https://anonymous.4open.science/w/ham-tts/)]
- [2024/01] **Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis** [[paper](https://openreview.net/forum?id=mvMI3N4AvD)][[demo](https://boostprompt.github.io/boostprompt/)]
- [2024/03] **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** [[paper](https://arxiv.org/abs/2403.03100v3)][[demo](https://speechresearch.github.io/naturalspeech3/)]
- [2024/01] **NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers** [[paper](https://openreview.net/forum?id=Rc7dAwVL3v)][[demo](https://speechresearch.github.io/naturalspeech2/)]
- [2024/03] **VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild** [[paper](https://arxiv.org/abs/2403.16973v2)][[code](https://github.com/jasonppy/VoiceCraft)][[demo](https://jasonppy.github.io/VoiceCraft_web/)] :heavy_check_mark:
- [2023/01] **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2301.02111v1)][[code](https://github.com/microsoft/unilm)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/)] *VALL-E* :heavy_check_mark:
- [2023/09] **PromptTTS 2: Describing and Generating Voices with Text Prompt** [[paper](https://arxiv.org/abs/2309.02285)][[code](https://github.com/microsoft/NeuralSpeech/tree/master/PromptTTS2)][[demo](https://speechresearch.github.io/prompttts2/)] :heavy_check_mark:
- [2023/09] **Matcha-tts: A fast tts architecture with conditional flow matching** [[paper](https://arxiv.org/abs/2309.03199v2)][[code](https://github.com/shivammehta25/Matcha-TTS)][[demo](https://shivammehta25.github.io/Matcha-TTS/)] :heavy_check_mark:
- [2023/09] **Voicebox: Text-guided multilingual universal speech generation at scale** [[paper](https://openreview.net/forum?id=gzCS252hCO)][[demo](https://voicebox.metademolab.com/)]
- [2023/09] **Voiceflow: Efficient text-to-speech with rectified flow matching** [[paper](https://arxiv.org/abs/2309.05027v2)][[code](https://github.com/X-LANCE/VoiceFlow-TTS)][[demo](https://cantabile-kwok.github.io/VoiceFlow/)] :heavy_check_mark:
- [2023/05] **Better speech synthesis through scaling** [[paper](https://arxiv.org/abs/2305.07243)][[code](https://github.com/neonbjb/tortoise-tts)] *tortoise-tts* :heavy_check_mark:


### Speech Language Models

- [WavChat](https://github.com/jishengpeng/WavChat) classify all spoken dialogue models based on whether **the core language model can directly understand and generate speech representations**, dividing them into cascaded and end-to-end categories.

#### End-to-End (Paradigms) Models

- [2025/04] **TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling** [[paper](https://arxiv.org/abs/2504.07053)][[code](https://github.com/mtkresearch/TASTE-SpokenLM)][[demo](https://mtkresearch.github.io/TASTE-SpokenLM.github.io/)] :heavy_check_mark:
- [2025/04] **VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation** [[paper](https://arxiv.org/abs/2504.04060)] *Code Comming Soon*
- [2025/03] **Qwen2.5-Omni Technical Report** [[paper](https://www.arxiv.org/abs/2503.20215)][[code](https://github.com/QwenLM/Qwen2.5-Omni)] :heavy_check_mark:
- [2025/02] **Conversational speech generation** [[blog](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo)][[code](https://github.com/SesameAILabs/csm)] :heavy_check_mark:
- [2025/02] **Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction** [[paper](https://www.arxiv.org/abs/2502.17239)][[code](https://github.com/baichuan-inc/Baichuan-Audio)] :heavy_check_mark:
- [2025/02] **Slamming: Training a Speech Language Model on One GPU in a Day** [[paper](https://arxiv.org/abs/2502.15814)][[code](https://github.com/slp-rl/slamkit)][[demo](https://pages.cs.huji.ac.il/adiyoss-lab/slamming/)] :heavy_check_mark:
- [2025/02] **ESPnet-SpeechLM: An Open Speech Language Model Toolkit** [[paper](https://arxiv.org/abs/2502.15218)][[code](https://github.com/espnet/espnet/tree/speechlm)] :heavy_check_mark:
- [2025/02] **Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction** [[paper](https://arxiv.org/abs/2502.11946)][[code](https://github.com/stepfun-ai/Step-Audio)] :heavy_check_mark:
- [2025/01] **Baichuan-Omni-1.5 Technical Report** [[paper](https://arxiv.org/abs/2501.15368)][[code](https://github.com/baichuan-inc/Baichuan-Omni-1.5)] *Baichuan-Omni-1.5* :heavy_check_mark:
- [2025/01] **A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone** [[blog](https://openbmb.notion.site/MiniCPM-o-2-6-GPT-4o-188ede1b7a558084b3aedd669cb80730)][[code](https://github.com/OpenBMB/MiniCPM-o)] *MiniCPM-o* :heavy_check_mark:
- [2025/01] **MinMo: A Multimodal Large Language Model for Seamless Voice Interaction** [[paper](https://arxiv.org/abs/2501.06282)][[demo](https://funaudiollm.github.io/minmo/)] *`Code Comming Soon`*
- [2025/01] **VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction** [[paper](https://arxiv.org/abs/2501.01957)][[code](https://github.com/VITA-MLLM/VITA)] :heavy_check_mark:
- [2025/01] **OmniChat: Enhancing Spoken Dialogue Systems with Scalable Synthetic Data for Diverse Scenarios** [[paper](https://arxiv.org/abs/2501.01384)][[demo](https://sharechatx.github.io/)] *including ShareChatX dataset*
- [2025/01] **SLIDE: Integrating Speech Language Model with LLM for Spontaneous Spoken Dialogue Generation** [[paper](https://arxiv.org/abs/2501.00805)][[demo](https://github.com/SpeechClub/SLIDE-demo/tree/main)]
- [2024/12] **Advancing Speech Language Models by Scaling Supervised Fine-Tuning with Over 60,000 Hours of Synthetic Speech Dialogue Data** [[paper](https://arxiv.org/abs/2412.01078)][[demo](https://huggingface.co/spaces/KE-Team/KE-Omni)] *Ke-SpeechChat & KE-Omni* | *`Code Comming Soon`*
- [2024/12] **Long-Form Speech Generation with Spoken Language Models** [[paper](https://arxiv.org/abs/2412.18603)][[demo](https://google.github.io/tacotron/publications/speechssm/)] *SpeechSSM, Long-Form Generation* 
- [2024/12] **SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training** [[paper](https://arxiv.org/abs/2412.15649)][[code](https://github.com/X-LANCE/SLAM-LLM)][[demo](https://slam-omni.github.io/)] :heavy_check_mark:
- [2024/12] **Continuous Speech Tokens Makes LLMs Robust Multi-Modality Learners** [[paper](https://arxiv.org/abs/2412.04917)][[demo](https://cognitivespeech.github.io/flowomni)] *Flow-Omni, continuous speech tokens*
- [2024/02] **Paralinguistics-Aware Speech-Empowered LLMs for Natural Conversation** [[paper](https://arxiv.org/abs/2402.05706)][[code](https://github.com/naver-ai/usdm)][[demo](https://unifiedsdm.github.io/)] *learning cross-modal distributional semantics* :heavy_check_mark:
- [2024/12] **GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot** [[paper](https://www.arxiv.org/abs/2412.02612)][[code](https://github.com/THUDM/GLM-4-Voice)] *speech interaction model & emotion, intonation, speech rate, and dialect & low latency* :heavy_check_mark:
- [2024/11] **MooER: Moore-threads Open Omni model for speech-to-speech intERaction** [[code](https://github.com/MooreThreads/MooER)] `Paper Comming Soon`
- [2024/11] **SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation** [[paper](https://arxiv.org/abs/2411.18138)] *Code Comming Soon* | *free-codec*
- [2024/11] **Building a Taiwanese Mandarin Spoken Language Model: A First Attempt** [[paper](https://arxiv.org/abs/2411.07111)][[code](https://github.com/nervjack2/SpeechChatGPTStreaming/)] *`Code Comming Soon`*
- [2024/11] **hertz-dev** [[code](https://github.com/Standard-Intelligence/hertz-dev)] :heavy_check_mark:
- [2024/11] **Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM** [[paper](https://arxiv.org/abs/2411.00774)][[demo](https://freeze-omni.github.io/)][[code](https://github.com/VITA-MLLM/Freeze-Omni)] *frozen llm in training* :heavy_check_mark:
- [2024/10] **Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities** [[paper](https://arxiv.org/abs/2410.11190)][[code](https://github.com/gpt-omni/mini-omni2)] :heavy_check_mark:
- [2024/10] **IntrinsicVoice: Empowering LLMs with Intrinsic Real-time Voice Interaction Abilities** [[paper](https://arxiv.org/abs/2410.08035)][[demo](https://instrinsicvoice.github.io/)] *reducing the length difference between speech and text*
- [2024/10] **OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation** [[paper](https://arxiv.org/abs/2410.17799)][[demo](https://omniflatten.github.io/)] *`Code Comming Soon`*
- [2024/09] **Westlake-Omni: Open-Source Chinese Emotional Speech Interaction Large Language Model with Unified Discrete Sequence Modeling** [[code](https://github.com/xinchen-ai/Westlake-Omni)] :heavy_check_mark:
- [2024/09] **Description-based Controllable Text-to-Speech with Cross-Lingual Voice Control** [[paper](https://arxiv.org/abs/2409.17452)][[demo](https://r9y9.github.io/projects/nansyttspp/)]
- [2024/09] **Moshi: a speech-text foundation model for real time dialogue** [[paper](https://kyutai.org/Moshi.pdf)][[code](https://github.com/kyutai-labs/moshi)][[demo](https://moshi.chat/)] *low delay* | *only english* :heavy_check_mark:
- [2024/09] **LLaMA-Omni: Seamless Speech Interaction with Large Language Models** [[paper](https://arxiv.org/abs/2409.06666)][[code](https://github.com/ictnlp/LLaMA-Omni)][[demo](https://replicate.com/ictnlp/llama-omni)] *only english* :heavy_check_mark:
- [2024/09] **EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions** [[paper](https://arxiv.org/abs/2409.18042)][[demo](https://emova-ollm.github.io/)]
- [2024/08] **Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming** [[paper](https://arxiv.org/abs/2408.16725)][[code](https://github.com/gpt-omni/mini-omni)] *End-to-End | speech interaction model* :heavy_check_mark:
- [2024/08] **Speech To Speech: an effort for an open-sourced and modular GPT4-o** [[code](https://github.com/huggingface/speech-to-speech)] *End-to-End | speech interaction model* :heavy_check_mark:
- [2024/08] **Language Model Can Listen While Speaking** [[paper](https://arxiv.org/abs/2408.02622)][[demo](https://ziyang.tech/LSLM/)] *Full Duplex Modeling | speech interaction model*
- [????/??] **SpeechGPT2: End-to-End Human-Like Spoken Chatbot** [[paper]()][[code](https://github.com/0nutation/SpeechGPT)][[demo](https://0nutation.github.io/SpeechGPT2.github.io/)] *paper & `Code Comming Soon` | speech interaction model*
- [2024/01] **SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation** [[paper](https://arxiv.org/abs/2401.13527)][[demo](https://0nutation.github.io/SpeechGPT-Gen.github.io/)] *`Code Comming Soon`*
- [2023/05] **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities** [[paper](https://arxiv.org/abs/2305.11000)][[code](https://github.com/0nutation/SpeechGPT)][[demo](https://0nutation.github.io/SpeechGPT.github.io/)] :heavy_check_mark:
- [2024/07] **Generative Expressive Conversational Speech Synthesis** [[paper](https://arxiv.org/abs/2407.21491)][[code](https://github.com/walker-hyf/GPT-Talker)] *GPT-Talker* ｜ *GPT for response and Style, VITS for audio* :heavy_check_mark:
- [2024/06] **GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities** [[paper](https://arxiv.org/abs/2406.11768)][[code](https://github.com/Sreyan88/GAMA)][[demo](https://sreyan88.github.io/gamaaudio/)] :heavy_check_mark:
- [2024/02] **Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities** [[paper](https://arxiv.org/abs/2402.01831)][[code](https://github.com/NVIDIA/audio-flamingo)][[demo](https://audioflamingo.github.io/)] :heavy_check_mark:
- [2024/02] **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** [[paper](https://arxiv.org/abs/2402.12226)][[code](https://github.com/OpenMOSS/AnyGPT)][[demo](https://junzhan2000.github.io/AnyGPT.github.io/)] :heavy_check_mark:
- [2024/03] **WavLLM: Towards Robust and Adaptive Speech Large Language Model** [[paper](https://arxiv.org/abs/2404.00656)][[code](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)] :heavy_check_mark:
- [2024/08] **DualSpeech: Enhancing Speaker-Fidelity and Text-Intelligibility Through Dual Classifier-Free Guidance** [[paper](https://arxiv.org/abs/2408.14423)][[demo](https://dualspeech.notion.site/DualSpeech-Demo-25fcf06ea94b4a739094d828d400542d)]
- [2024/08] **Style-Talker: Finetuning Audio Language Model and StyleBased Text-to-Speech Model for Fast Spoken Dialogue Generation** [[paper](https://arxiv.org/abs/2408.11849)][[code](https://github.com/xi-j/Style-Talker)][[demo](https://styletalker.github.io/)] :heavy_check_mark:
- [2024/04] **CoVoMix: Advancing Zero-Shot Speech Generation for Human-like Multi-talker Conversations** [[paper](https://arxiv.org/abs/2404.06690)][[code](https://github.com/vivian556123/NeurIPS2024-CoVoMix)][[demo](https://www.microsoft.com/en-us/research/project/covomix/)] *multi-round dialogue speech generation* :heavy_check_mark:

#### Cascaded (Paradigms) Models 

- [2025/03] **Mellow: a small audio language model for reasoning** [[paper](https://arxiv.org/abs/2503.08540)][[code](https://github.com/soham97/mellow)] :heavy_check_mark:
- [2024/11] **A fast multimodal LLM for real-time voice** [[blog](https://www.ultravox.ai/blog/ultravox-an-open-weight-alternative-to-gpt-4o-realtime)][[code](https://github.com/fixie-ai/ultravox)][[demo](https://demo.ultravox.ai/)] *Ultravox* :heavy_check_mark:
- [2024/10] **Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant** [[paper](https://arxiv.org/abs/2410.15316)][[code](https://github.com/janhq/ichigo)] :heavy_check_mark:
- [2024/10] **Ocean-omni: To Understand the World with Omni-modality** [[paper](https://arxiv.org/abs/2410.08565)][[code](https://github.com/westlake-baichuan-mllm/bc-omni)] *Baichuan-Omni* :heavy_check_mark:
- [2024/08] **VITA: Towards Open-Source Interactive Omni Multimodal LLM** [[paper](https://www.arxiv.org/abs/2408.05211)][[code](https://github.com/VITA-MLLM/VITA)][[demo](https://vita-home.github.io/)] :heavy_check_mark:
- [2024/07] **Qwen2-Audio Technical Report** [[paper](https://www.arxiv.org/abs/2407.10759)][[code](https://github.com/QwenLM/Qwen2-Audio)] :heavy_check_mark:
- [2024/05] **A Full-duplex Speech Dialogue Scheme Based On Large Language Model** [[paper](https://arxiv.org/abs/2405.19487)] *neural finite state machine* 
- [2023/11] **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** [[paper](https://arxiv.org/abs/2311.07919)][[code](https://github.com/QwenLM/Qwen-Audio)] :heavy_check_mark:
- [2023/10] **SALMONN: Towards Generic Hearing Abilities for Large Language Models** [[paper](https://arxiv.org/abs/2310.13289)][[code](https://github.com/bytedance/SALMONN)] :heavy_check_mark:

### Benchmark

- [2025/03] **Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities** [[paper](https://www.arxiv.org/abs/2503.04721)][[code](https://github.com/DanielLin94144/Full-Duplex-Bench)]
- [2025/03] **Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics** [[paper](https://arxiv.org/abs/2503.01174)] *Code Comming Soon*
- [2025/02] **URO-Bench: A Comprehensive Benchmark for End-to-End Spoken Dialogue Models** [[paper](https://www.arxiv.org/abs/2502.17810)]
- [2023/09] **Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech** [[paper](https://arxiv.org/abs/2309.09510)]
- [2024/02] **Codec-SUPERB: An In-Depth Analysis of Sound Codec Models** [[paper](https://arxiv.org/abs/2402.13071v2)][[code](https://github.com/voidful/Codec-SUPERB)]
- [2024/07] **EMO-Codec: A Depth Look at Emotion Preservation Capacity of Legacy and Neural Codec Models With Subjective and Objective Evaluations** [[paper](https://arxiv.org/abs/2407.15458)]
- [2024/06] **DASB - Discrete Audio and Speech Benchmark** [[paper](https://arxiv.org/abs/2406.14294)][[code](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/DASB)] *a benchmark for evaluating discrete audio representations*
- [2024/12] **Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models** [[paper](https://arxiv.org/abs/2412.05167)]


### Survey

- [2025/04] **On The Landscape of Spoken Language Models: A Comprehensive Survey** [[paper](https://arxiv.org/abs/2504.08528)]
- [2025/02] **Recent Advances in Discrete Speech Tokens: A Review** [[paper](https://arxiv.org/abs/2502.06490)]
- [2024/12] **Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey** [[paper](https://www.arxiv.org/abs/2412.18619)][[code](https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction)]
- [2024/12] **Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey** [[paper](https://arxiv.org/abs/2412.06602)][[code](https://github.com/imxtx/awesome-controllabe-speech-synthesis)]
- [2024/11] **WavChat: A Survey of Spoken Dialogue Models** [[paper](https://www.arxiv.org/abs/2411.13577)][[code](https://github.com/jishengpeng/WavChat)]
- [2024/10] **Recent Advances in Speech Language Models: A Survey** [[paper](https://arxiv.org/abs/2410.03751)]
- [2024/10] **A Survey on Speech Large Language Models** [[paper](https://arxiv.org/abs/2410.18908)]
- [2024/02] **Towards audio language modeling -- an overview** [[paper](https://arxiv.org/abs/2402.13236)]


## Some Interesting Models

- [2025/04] **Dopamine Audiobook: A Training-free MLLM Agent for Emotional and Human-like Audiobook Generation** [[paper](https://arxiv.org/abs/2504.11002)][[demo](https://dopamine-audiobook.github.io/)]
- [2025/03] **WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching** [[paper](https://www.arxiv.org/abs/2503.16689)][[code](https://github.com/luotianze666/WaveFM)][[demo](https://luotianze666.github.io/WaveFM/)] :heavy_check_mark:
- [2025/03] **Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering** [[paper](https://arxiv.org/abs/2503.11197)][[code](https://github.com/xiaomi-research/r1-aqa)] :heavy_check_mark:
- [2025/03] **FilmComposer: LLM-Driven Music Production for Silent Film Clips** [[paper](https://www.arxiv.org/abs/2503.08147)][[code](https://github.com/Apple-jun/FilmComposer)][[demo](https://apple-jun.github.io/FilmComposer.github.io/)] *Code Comming Soon*
- [2023/07] **WavJourney: Compositional Audio Creation with Large Language Models** [[paper](https://arxiv.org/abs/2307.14335)][[code](https://github.com/Audio-AGI/WavJourney)][[demo](https://audio-agi.github.io/WavJourney_demopage/)] :heavy_check_mark:
- [2025/03] **PodAgent: A Comprehensive Framework for Podcast Generation** [[paper](https://arxiv.org/abs/2503.00455)][[code](https://github.com/yujxx/PodAgent)][[demo](https://podcast-agent.github.io/demo/)] :heavy_check_mark:
- [2024/09] **Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects** [[paper](https://arxiv.org/abs/2409.18847)][[code](https://github.com/anniejchu/text2fx)][[demo](https://anniejchu.github.io/text2fx/)] :heavy_check_mark:
- [2025/02] **DisCoder: High-Fidelity Music Vocoder Using Neural Audio Codecs** [[paper](https://arxiv.org/abs/2502.12759)][[code](https://github.com/ETH-DISCO/discoder/)][[demo](https://lucala.github.io/discoder/)] :heavy_check_mark:
- [2024/08] **PeriodWave: Multi-Period Flow Matching for High-Fidelity Waveform Generation** [[paper](https://arxiv.org/abs/2408.07547)][[code](https://github.com/sh-lee-prml/PeriodWave)][[demo](https://periodwave.github.io/demo/)] :heavy_check_mark:
- [2025/02] **Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment** [[paper](https://arxiv.org/abs/2502.04328)][[code](https://github.com/Ola-Omni/Ola)][[demo](https://ola-omni.github.io/)] :heavy_check_mark:
- [2024/06] **Scaling up masked audio encoder learning for general audio classification** [[paper](https://arxiv.org/abs/2406.06992)][[code](https://github.com/richermans/dasheng/)] *DaSheng* :heavy_check_mark:
- [2025/02] **Metis: A Foundation Speech Generation Model with Masked Generative Pre-training** [[paper](https://arxiv.org/abs/2502.03128)][[code](https://github.com/open-mmlab/Amphion)][[demo](https://metis-demo.github.io/)] :heavy_check_mark:
- [2025/01] **CodecFake-Omni: A Large-Scale Codec-based Deepfake Speech Dataset** [[paper](https://arxiv.org/abs/2501.08238)] *`Code Comming Soon`*
- [2024/12] **TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization** [[paper](https://arxiv.org/abs/2412.21037)][[code](https://github.com/declare-lab/TangoFlux)][[demo](https://tangoflux.github.io/)] :heavy_check_mark:
- [2024/12] **Flow Matching Guide and Code** [[paper](https://arxiv.org/abs/2412.06264)][[code](https://github.com/facebookresearch/flow_matching)]
- [2024/12] **OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows** [[paper](https://arxiv.org/abs/2412.01169)][[code](https://github.com/jacklishufan/OmniFlows)] :heavy_check_mark:
- [2024/09] **SSR-Speech: Towards Stable, Safe and Robust Zero-shot Text-based Speech Editing and Synthesis** [[paper](https://arxiv.org/abs/2409.07556)][[code](https://github.com/WangHelin1997/SSR-Speech)][[demo](https://wanghelin1997.github.io/SSR-Speech-Demo/)] :heavy_check_mark:
- [2024/11] **FLowHigh: Towards efficient and high-quality audio super-resolution with single-step flow matching** [[code](https://github.com/jjunak-yun/FLowHigh_code)][[demo](https://jjunak-yun.github.io/FLowHigh/)] :heavy_check_mark:
- [2024/11] **O1 Replication Journey: A Strategic Progress Report -- Part 1** [[paper](https://arxiv.org/abs/2410.18982)][[code](https://github.com/GAIR-NLP/O1-Journey/)] :heavy_check_mark:
- [2024/11] **LLaMA-O1: Open Large Reasoning Model Frameworks For Training, Inference and Evaluation With PyTorch and HuggingFace** [[code](https://github.com/SimpleBerry/LLaMA-O1)] :heavy_check_mark:
- [2024/07] **Speech-Copilot: Leveraging Large Language Models for Speech Processing via Task Decomposition, Modularization, and Program Generation** [[paper](https://arxiv.org/abs/2407.09886)][[code](https://github.com/kuan2jiu99/speech-copilot)][[demo](https://sites.google.com/view/slt2024-demo-page)] :heavy_check_mark:
- [2024/07] **Stable Audio Open** [[paper](https://arxiv.org/abs/2407.14358)] [[code](https://huggingface.co/stabilityai/stable-audio-open-1.0)] :heavy_check_mark:
- [2024/05] **EmoLLM(心理健康大模型)** [[code](https://github.com/SmartFlowAI/EmoLLM/blob/main/generate_data/tutorial.md)][[demo](https://openxlab.org.cn/apps/detail/Farewell1/EmoLLMV2.0)] :heavy_check_mark:
- [2023/02] **Improving and generalizing flow-based generative models with minibatch optimal transport** [[paper](https://arxiv.org/abs/2302.00482)][[code](https://github.com/atong01/conditional-flow-matching)] *TorchCFM* | *Tutorials* :heavy_check_mark:
- [2022/10] **Flow Matching for Generative Modeling** [[paper](https://arxiv.org/abs/2210.02747)] *Conditional Flow Matching*
- [2022/09] **Rectified Flow: A Marginal Preserving Approach to Optimal Transport** [[paper](https://arxiv.org/abs/2209.14577)][[code](https://github.com/gnobitab/RectifiedFlow)] :heavy_check_mark:


## Music Generation

- [2025/04] **MusFlow: Multimodal Music Generation via Conditional Flow Matching** [[paper](https://arxiv.org/abs/2504.13535)][[demo](https://anonymous22356.github.io/musflow.github.io/)] *Code Comming Soon*
- [2025/04] **A Survey on Music Generation from Single-Modal, Cross-Modal, and Multi-Modal Perspectives: Data, Methods, and Challenges** [[paper](https://arxiv.org/abs/2504.00837)]
- [2025/03] **Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation** [[paper](https://arxiv.org/abs/2503.19611)][[demo](https://musicot.github.io/)] *MusiCoT*
- [2025/03] **DiffRhythm: Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation with Latent Diffusion** [[paper](https://arxiv.org/abs/2503.01183)][[code](https://github.com/ASLP-lab/DiffRhythm)][[demo](https://nzqian.github.io/DiffRhythm/)] :heavy_check_mark:
- [2025/02] **SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation** [[paper](https://arxiv.org/abs/2502.13128)][[code](https://github.com/LiuZH-19/SongGen)][[demo](https://liuzh-19.github.io/SongGen/)] :heavy_check_mark:
- [2025/02] **TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching** [[paper](https://arxiv.org/abs/2502.12572)][[code](https://github.com/gwx314/TechSinger)][[demo](https://tech-singer.github.io/)] :heavy_check_mark:
- [2025/02] **CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages** [[paper](https://arxiv.org/abs/2502.10362)][[code](https://github.com/sanderwood/clamp3)][[demo](https://sanderwood.github.io/clamp3/)] :heavy_check_mark:
- [2025/02] **YuE: Open Music Foundation Models for Full-Song Generation** [[paper](https://www.arxiv.org/abs/2503.08638)][[code](https://github.com/multimodal-art-projection/YuE)][[demo](https://map-yue.github.io/)] :heavy_check_mark:
- [2025/01] **InspireMusic: A Unified Framework for Music, Song, Audio Generation** [[paper](https://www.arxiv.org/abs/2503.00084)][[code](https://github.com/FunAudioLLM/InspireMusic)][[demo](https://iris2c.github.io/InspireMusic/)] :heavy_check_mark:
- [2024/12] **SongEditor: Adapting Zero-Shot Song Generation Language Model as a Multi-Task Editor** [[paper](https://www.arxiv.org/abs/2412.13786)][[demo](https://cypress-yang.github.io/SongEditor_demo/)]
- [2024/12] **MuMu-LLaMA: Multi-modal Music Understanding and Generation via Large Language Models** [[paper](https://arxiv.org/abs/2412.06660)][[code](https://github.com/shansongliu/MuMu-LLaMA)] :heavy_check_mark:
- [2024/10] **MusicFlow: Cascaded Flow Matching for Text Guided Music Generation** [[paper](https://arxiv.org/abs/2410.20478v1)] *`Code Comming Soon`* | *Similar to MaskGCT*
- [2024/09] **FLUX that Plays Music** [[paper](https://arxiv.org/abs/2409.00587)][[code](https://github.com/feizc/FluxMusic)][[melodio](https://www.melodio.ai/)] *KunLun* :heavy_check_mark:
- [2024/09] **Seed-Music: A Unified Framework for High Quality and Controlled Music Generation** [[paper](https://arxiv.org/abs/2409.09214)][[demo](https://team.doubao.com/en/special/seed-music)] *tech-report*
- [2024/05] **QA-MDT: Quality-aware Masked Diffusion Transformer for Enhanced Music Generation** [[paper](https://arxiv.org/abs/2405.15863)][[code](https://github.com/ivcylc/qa-mdt)][[demo](https://qa-mdt.github.io/)] :heavy_check_mark:
- [2024/05] **Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning** [[paper](https://arxiv.org/abs/2405.18386v2)][[code](https://github.com/ldzhangyx/instruct-MusicGen)][[demo](https://foul-ice-5ea.notion.site/Instruct-MusicGen-Demo-Page-a1e7d8d474f74df18bda9539d96687ab)] *Instruction Tuning* :heavy_check_mark:
- [2023/06] **Simple and Controllable Music Generation** [[paper](https://arxiv.org/abs/2306.05284)][[code](https://github.com/facebookresearch/audiocraft)] *Prompt Control | AudioCraft* :heavy_check_mark:

## Speech DataSet

- [2025/04] **SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning** [[paper](https://arxiv.org/abs/2504.09081)]
- [2025/03] **Scaling Rich Style-Prompted Text-to-Speech Datasets** [[paper](https://www.arxiv.org/abs/2503.04713)][[code](https://github.com/ajd12342/paraspeechcaps)][[demo](https://paraspeechcaps.github.io/)] :heavy_check_mark:
- [2025/02] **CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition** [[paper](https://www.arxiv.org/abs/2502.18913)]
- [2025/02] **Audio-FLAN: A Preliminary Release** [[paper](https://www.arxiv.org/abs/2502.16584)][[code](https://github.com/lmxue/Audio-FLAN)][[dataset](https://huggingface.co/datasets/HKUSTAudio/Audio-FLAN-Dataset)] :heavy_check_mark:
- [2024/08] **SpeechCraft: A Fine-grained Expressive Speech Dataset with Natural Language Description** [[paper](https://arxiv.org/abs/2408.13608)][[code](https://github.com/thuhcsi/SpeechCraft)][[demo](https://speechcraft2024.github.io/speechcraft2024/)] :heavy_check_mark:
- [2024/07] **Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation** [[paper](https://arxiv.org/abs/2407.05361)][[code](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)][[demo](https://emilia-dataset.github.io/Emilia-Demo-Page/)][[dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)] :heavy_check_mark:
- [2024/06] **WenetSpeech4TTS: A 12,800-hour Mandarin TTS corpus for large speech generation model benchmark** [[paper](https://arxiv.org/abs/2406.05763)][[demo](https://wenetspeech4tts.github.io/wenetspeech4tts/)][[dataset](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS)] :heavy_check_mark:
- [2020/10] **Didispeech: A large scale Mandarin speech corpus** [[paper](https://arxiv.org/abs/2010.09275)][[code](https://github.com/athena-team/DiDiSpeech)][[demo](https://athena-team.github.io/DiDiSpeech/)][[dataset](??)]


## Some Interesting knowledge

### Blog & Courses

- **Anthropic courses** [[github](https://github.com/anthropics/courses)]
- **LLM101n: Let's build a Storyteller** [[github](https://github.com/karpathy/LLM101n)]
- **Build a Large Language Model (From Scratch)** [[github](https://github.com/rasbt/LLMs-from-scratch)]
- **build nanoGPT from Karpathy** [[github](https://github.com/karpathy/build-nanogpt)]


### Minor Points of Concern

<details>
<summary>GitHub</summary>
 
- ChatTTS: https://github.com/2noise/ChatTTS/tree/main
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- Bert-vits2-NoBug: https://github.com/ywh-my/Bert-vits2-NoBug
- VoiceCraft: https://github.com/jasonppy/VoiceCraft
- YourTTS: https://github.com/Edresson/YourTTS
- Coqui: https://github.com/coqui-ai/TTS
- ebook2audiobookXTTS: https://github.com/DrewThomasson/ebook2audiobookXTTS
- MARS5-TTS: https://github.com/Camb-ai/MARS5-TTS
- edge-tts: https://github.com/rany2/edge-tts
- metavoice-src: https://github.com/metavoiceio/metavoice-src
- StyleTTS2: https://github.com/yl4579/StyleTTS2
- open-tts-tracker: https://github.com/Vaibhavs10/open-tts-tracker
- Amphion: https://github.com/open-mmlab/Amphion
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- CFM: https://github.com/atong01/conditional-flow-matching
- speech-trident: https://github.com/ga642381/speech-trident
- bark: https://github.com/suno-ai/bark
- LangGPT: https://github.com/langgptai/LangGPT (提示词工程)
- composio: https://github.com/ComposioHQ/composio
- torchdiffeq: https://github.com/rtqichen/torchdiffeq
- podlm: https://github.com/lihuithe/podlm-public (NoteBookLM 的平替)
- NotebookLlama: https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama (类似 NoteBookLM)
- playnote: https://play.ai/playnote (类似 NotebookLM)
- podcastfy: https://github.com/souzatharsis/podcastfy (类似 NotebookLM)
- dify: https://github.com/langgenius/dify (开源的 LLM 应用开发平台)
- Awesome-Dify-Workflow: https://github.com/svcvit/Awesome-Dify-Workflow
- LiblibAI: https://www.liblib.art (AI创作平台)
</details>

<details>
<summary>Nice Tool</summary>
 
- pytorch-OpCounter: https://github.com/Lyken17/pytorch-OpCounter
- rich: https://github.com/Textualize/rich
- argbind: https://github.com/pseeth/argbind/
- audiotools: https://github.com/descriptinc/audiotools
- hydra: https://github.com/facebookresearch/hydra
- joblib: https://github.com/joblib/joblib
- einops: https://github.com/arogozhnikov/einops
- safetensors: https://github.com/huggingface/safetensors
- OpenDiloco: https://github.com/PrimeIntellect-ai/OpenDiloco
- WeTextProcessing: https://github.com/wenet-e2e/WeTextProcessing
- zed: https://github.com/zed-industries/zed
- weekly: https://github.com/ljinkai/weekly
- tinygrad: https://github.com/tinygrad/tinygrad
- ffmpeg-normalize: https://github.com/slhck/ffmpeg-normalize
- kohya_ss: https://github.com/bmaltais/kohya_ss
- Lora-Training-in-Comfy: https://github.com/LarryJane491/Lora-Training-in-Comfy
- ComfyUI-Manager: https://github.com/ltdrdata/ComfyUI-Manager
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- comfyui-workspace-manager: https://github.com/11cafe/comfyui-workspace-manager
- CosyVoice+ComfyUI: https://github.com/AIFSH/CosyVoice-ComfyUI
- ComfyUI-wiki: https://github.com/602387193c/ComfyUI-wiki
- ZHO: https://github.com/ZHO-ZHO-ZHO
- tmux: https://github.com/tmux/tmux
- LoRAlib: https://github.com/microsoft/LoRA
- codespaces: https://github.com/codespaces
- Foliate(PDF): https://johnfactotum.github.io/foliate/
- Okular(PDF): https://okular.kde.org/zh-cn/
- audioFlux: https://github.com/libAudioFlux/audioFlux
- PyWavelets: https://github.com/PyWavelets/pywt
- 智能体或工作流平台: https://ai-bot.cn/ai-agent-development-platform/
- open-webui: https://github.com/open-webui/open-webui
- qwen-2.5-code-interpreter: https://github.com/cfahlgren1/qwen-2.5-code-interpreter
- ollama: https://github.com/ollama/ollama; https://ollama.com/
- vllm: https://github.com/vllm-project/vllm
- anythingLLM: https://github.com/Mintplex-Labs/anything-llm
- Windsurf: https://codeium.com/windsurf
- cursor: https://www.cursor.com/
- docling: https://github.com/DS4SD/docling
- TEN-Agent: https://github.com/TEN-framework/TEN-Agent
</details>

## Reference

- [ZhiHu][别慌！一文教你看懂GPT-4o背后的语音技术](https://zhuanlan.zhihu.com/p/698725358)
- [ZhiHu][百花齐放的Audio Codec: 语音合成利器](https://zhuanlan.zhihu.com/p/696434090)
- [InterSpeech2024][InterSpeech2024 Speech Processing Using Discrete Speech Units](https://interspeech2024.org/special-sessions-challenges/) : https://www.wavlab.org/activities/2024/Interspeech2024-Discrete-Speech-Unit-Challenge/ : https://huggingface.co/discrete-speech : [arxiv 2024](https://arxiv.org/abs/2406.07725) : [[paper](https://www.isca-archive.org/interspeech_2024/index.html)]
- [Slides][Challenges in Developing Spoken Language Models](https://drive.google.com/file/d/1gPjnjGKxeCF72gisPVuQlDvogXQCtNk4/view) *slides*
- [GitHub][speech-trident](https://github.com/ga642381/speech-trident) : Awesome speech/audio LLMs, representation learning, and codec models
- [GitHub][Awesome-Speech-Language-Model](https://github.com/ddlBoJack/Awesome-Speech-Language-Model) : Paper, Code and Resources for Speech Language Model and End2End Speech Dialogue System



