## NLI-Based Hallucination Detection for RAG: Best Practices and Production Strategies

Your observation about the high neutral scores and low contradiction probabilities is **standard behavior for pre-trained, zero-shot NLI models** and reveals a fundamental limitation in how these models generalize from their training distribution. This is not a bug—it's a feature gap that production RAG systems address through systematic fine-tuning, threshold strategies, and architectural choices. Here's what the research and production systems actually do:

### Question 1: Treating High Neutral Scores as "Supported"

**The short answer: No, this defeats the purpose and is not standard practice.**

Using high neutral scores (>50%) with low contradiction (<20%) as "supported" collapses the NLI classification into a binary entailment-vs-not-entailment decision, which destroys the discriminative signal that makes NLI valuable in the first place. However, this observation points to a critical insight: your zero-shot model is underfitted for the RAG task.[^1_1]

The research literature and production systems (Luna, LettuceDetect, HHEM) handle this through **explicit label remapping during fine-tuning**. In the SciHal benchmark competition, researchers explicitly mapped the NLI neutral label to "unverifiable" (unsupported) rather than treating it as a middle ground. Similarly, in production implementations, neutral predictions are systematically treated as unsupported during threshold tuning, not as ambiguous cases.[^1_2][^1_3]

**The fundamental problem with your current setup**: Pre-trained NLI models trained on SNLI/MultiNLI/FEVER use neutral to represent logical indeterminacy (statements that could be true but aren't implied). In RAG hallucination detection, you need neutral to represent "semantic gap" or "insufficient lexical alignment"—a completely different phenomenon. Without fine-tuning, the model defaults to SNLI/MultiNLI behavior.

### Question 2: How Production RAG Systems Achieve ~0.83 F1

The 0.83 F1 figure you referenced comes from the **RAGTruth benchmark** where fine-tuned Llama-2-13B achieved 78.7% F1 on the response-level task, and more recent approaches (RAG-HAT using Llama-3-8B with DPO fine-tuning) have exceeded this. Here's what separates production systems from zero-shot approaches:[^1_4][^1_5]

**Strategy 1: Fine-tuned NLI Models on RAG-Specific Data**

This is the dominant approach in 2024-2025 production systems. The key variants:

- **Luna** (DeBERTA-large, 440M parameters) fine-tuned on curated real-world RAG data achieves 65.4% F1 on RAGTruth—competitive with zero-shot GPT-3.5. It specifically handles **long-context RAG** (a major challenge for generic NLI), maintaining 88-68% of performance on 5k-16k+ token contexts.[^1_6]
- **LettuceDetect** (ModernBERT, 330M parameters) achieves 79.22% F1—a 14.8% improvement over Luna—by using ModernBERT's 8,192-token context window and training on the RAGTruth corpus. This demonstrates that architecture matters: ModernBERT's local-global attention handles long-form RAG verification better than DeBERTA.[^1_7]
- **DeBERTa-v3-large** pre-trained on five diverse NLI corpora (SNLI, MultiNLI, FEVER, ANLI, LING-WANLI) **without any fine-tuning** ranked fourth on the SciHal leaderboard, outperforming more complex feature-based pipelines. This suggests that DeBERTA's pre-training is exceptionally strong, but fine-tuning on task-specific data is necessary for top performance.[^1_2]

**Strategy 2: Dual-Tier Filtering with Verifiability Classification**

Production systems don't treat all statements equally. **RAGHalu** uses a two-tiered approach:[^1_8]

1. **Tier 1**: Classify each statement as VERIFIABLE or NO-INFO (statements like "we'll look into that" aren't factual claims)
2. **Tier 2**: Only for VERIFIABLE statements, use NLI to classify as SUPPORTED or UNSUPPORTED

This achieves **0.93 F1 on UNSUPPORTED claim detection** on production brand data and **0.96-0.97 F1** on banking/credit union datasets. The filteringStep dramatically improves signal-to-noise.[^1_8]

**Strategy 3: Threshold Tuning and Calibration**

The ~0.83 F1 baseline isn't achieved with raw softmax scores. Production implementations:

- Use **probabilistic calibration** on validation data to find optimal thresholds for entailment vs. neutral vs. contradiction. A production implementation reported that threshold of **0.8 for entailment** optimized accuracy on their trial set.[^1_3]
- **Ensemble multiple detectors** (hallucination-detection model + NLI model + embedding-based similarity) and use a voting classifier, which achieves 77.8-79.9% accuracy depending on domain.[^1_3]
- Map neutral scores explicitly: neutral at >50% + contradiction <20% typically becomes UNSUPPORTED (not SUPPORTED).

**Strategy 4: Combine NLI with Semantic Similarity**

The research shows that **NLI alone has systematic weaknesses**. Production systems combine:[^1_9]

- NLI entailment scores
- Semantic similarity scores (embedding-based)
- Token-overlap metrics (BLEU/ROUGE/Jaccard)

The paper comparing hallucination detectors found that **NLI contradiction scoring** is more reliable than semantic similarity alone, which exhibits very low accuracy (often near random on some datasets). However, the combination is stronger than either alone.[^1_9]

### Question 3: Recommended NLI Models and Configurations (2024-2025)

| **Model** | **Architecture** | **Best For** | **Key Advantage** | **Citation** |
| :-- | :-- | :-- | :-- | :-- |
| **DeBERTa-v3-large** (Tasksource NLI checkpoint) | Encoder, 434M | General-purpose, scientific hallucination detection | Pre-trained on 5 diverse NLI corpora; ranked 4th on SciHal without fine-tuning | [^1_2] |
| **Luna (DeBERTA-large + RAGBench)** | Encoder, 440M | Production RAG systems, long-context (5k-16k tokens) | Handles long-context; 97% cost reduction vs. GPT-3.5; generalizes across domains | [^1_6] |
| **LettuceDetect (ModernBERT)** | Encoder, 330M | Long-form RAG (up to 8,192 tokens) | State-of-the-art on RAGTruth (79.22% F1); hardware-aware; efficient inference | [^1_7] |
| **HHEM-2.1-Open (Vectara)** | Encoder, 440M | Production hallucination detection | Unlimited context (vs. HHEM-1.0's 512-token limit); outperforms GPT-3.5/GPT-4 on three benchmarks | [^1_10] |
| **RoBERTa-large-mnli** | Encoder, 355M | Banking/credit union domains, RAGHalu tier 2 | F1 0.97 on credit union data; strong on production brand data | [^1_8] |
| **Longformer (HAT)** | Encoder + long attention, 435M | Long-context verification (16k+ tokens) | Designed for long-form; but shows high recall/low precision (overpredicts hallucinations) | [^1_11] |

**Configuration recommendations:**

- **Context window is critical**: Generic NLI models plateau at 512-1024 tokens. For modern RAG (which retrieves multi-document passages), use ModernBERT (8,192 tokens) or DeBERTA + chunking strategies.
- **Fine-tuning matters more than scale**: LettuceDetect (330M ModernBERT) outperforms Luna (440M DeBERTA), and both outperform zero-shot GPT-3.5. The relationship is: **[task-specific fine-tuning] > [pre-training scale]** in this domain.[^1_7]
- **Asymmetric label mapping is standard**: DeBERTA NLI pre-training maps neutral→unsupported explicitly; don't treat neutral as ambiguous.[^1_3]
- **Two-tier filtering is a best practice**: Implementing verifiability classification first (Tier 1) before NLI verification (Tier 2) increases F1 by ~5-10% in production systems.[^1_8]


### Question 4: Alternative Approaches (AlignScore, TRUE, G-Eval, etc.)

**The landscape of specialized hallucination detection:**

**AlignScore**: A semantic-alignment-based factual consistency metric trained on alignment tasks. However, research shows **AlignScore struggles with long-context inputs**—it exhibits "insufficient expressive capacity to distinguish hallucinations" on long-form text, with precision/recall of 50.09%/60.00%. The decompose-then-verify approach it uses (breaking text into claims and verifying each) **breaks down at scale** because aggregating claim-level scores doesn't capture interaction effects.[^1_11][^1_2][^1_9]

**TRUE (T5-based)**: A T5 model trained on NLI mixture. In head-to-head comparison, **google/t5_xxl_true_nli_mixture outperformed all other models on the Bank test set (F1 0.96) but struggled on others**—indicating severe dataset shift issues. TRUE is better viewed as a specialized baseline than a production solution.[^1_8]

**G-Eval / LLM-as-Judge**: Zero-shot GPT-4o/GPT-4o-mini with chain-of-thought prompting achieves **0.760 F1 (GPT-4o) and 0.734 F1 (GPT-4o-mini) on hallucination detection**—competitive with fine-tuned encoders but at 10-100x higher latency and cost. These are useful **when you need explanations**, but for high-throughput detection, fine-tuned encoders dominate.[^1_9]

**LUMINA** (2025): A novel framework decomposing hallucination detection into **external context utilization + internal knowledge utilization scores**. Achieves >0.9 AUROC on HalluRAG dataset—notably, focuses on how much the LLM *uses* retrieved context rather than just verifying claims. This shifts the detection signal from "does the claim match the context" to "did the LLM rely on hallucination internal knowledge instead of context."[^1_12]

**MiniCheck**: Synthesizes hallucinated examples using GPT-4 for training data augmentation. Competitive with fine-tuned models but requires fewer labeled examples. Good for low-data regimes.[^1_13]

**Production verdict (2024-2025):**

1. **For high-throughput RAG**: Use **LettuceDetect (ModernBERT) or Luna (DeBERTA)** fine-tuned on your domain data. If you have <5k labeled examples, fine-tune DeBERTA-v3-large with task-specific data.
2. **For long-context RAG** (16k+ tokens): LettuceDetect or Longformer-based approaches; avoid AlignScore.
3. **For maximum accuracy with explanations**: Use two-tier approach: (1) LLM-based verifiability filtering, (2) fine-tuned NLI or LLM-as-judge verification. This hybrid achieves 0.83-0.86 F1 in practice.[^1_8]
4. **Don't use AlignScore or TRUE as primary detectors** unless your context is <512 tokens; they exhibit systematic performance degradation on RAG-scale inputs.[^1_11]

***

## Practical Implementation Path

Given your goal of ~0.83 F1 and current struggle with neutral scores:

1. **Immediate**: Fine-tune `cross-encoder/nli-deberta-v3-base` on a subset of RAGTruth or your domain data. Use **explicit label mapping** (neutral→unsupported). This alone will likely improve F1 by 15-20%.
2. **Short-term**: Implement two-tier filtering—add a verifiability classifier (can be a simple fine-tuned RoBERTa binary classifier) before the NLI stage.
3. **Medium-term**: Experiment with **ModernBERT** if your contexts exceed 1k tokens; the long-context advantage is significant.
4. **Threshold tuning**: Systematically sweep threshold values on a held-out validation set; optimal thresholds are often 0.7-0.8 for entailment (not 0.5).
5. **Ensemble**: Combine your NLI detector with semantic similarity (e.g., sentence-transformers) or embedding-based methods for robustness.

The 0.83 F1 target is achievable with task-specific fine-tuning on 2-5k labeled examples and proper threshold calibration—but zero-shot NLI models **fundamentally cannot** reach this performance due to label distribution mismatch and the neutral-score problem you've identified.
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2504.15771v1

[^1_2]: https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use

[^1_3]: https://openreview.net/pdf/8be99a6d967dcddec7d5ba5e7c4e79238e435881.pdf

[^1_4]: https://aclanthology.org/2025.sdp-1.34.pdf

[^1_5]: https://arxiv.org/html/2406.00975v2

[^1_6]: https://arxiv.org/pdf/2505.21786.pdf

[^1_7]: https://arxiv.org/html/2502.17125v1

[^1_8]: https://arxiv.org/html/2410.08764v1

[^1_9]: https://aclanthology.org/2024.acl-long.585.pdf

[^1_10]: https://towardsdatascience.com/lettucedetect-a-hallucination-detection-framework-for-rag-applications/

[^1_11]: https://arxiv.org/html/2504.19457v1

[^1_12]: https://www.arxiv.org/pdf/2509.21875.pdf

[^1_13]: https://arxiv.org/html/2505.04847v2

[^1_14]: https://arxiv.org/html/2406.00975v1

[^1_15]: https://repositum.tuwien.at/bitstream/20.500.12708/219681/1/Verdha Nadia - 2025 - Multilingual Hallucination Detection for RAG applications.pdf

[^1_16]: https://aclanthology.org/2024.emnlp-industry.2.pdf

[^1_17]: https://aclanthology.org/2025.sdp-1.33.pdf

[^1_18]: https://aclanthology.org/2025.findings-emnlp.1035.pdf

[^1_19]: https://towardsdatascience.com/how-to-perform-hallucination-detection-for-llms-b8cb8b72e697/

[^1_20]: https://arxiv.org/html/2410.22977v1

[^1_21]: https://arxiv.org/html/2509.03531v1

[^1_22]: https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/

[^1_23]: https://arxiv.org/html/2508.17127

[^1_24]: https://aclanthology.org/2024.emnlp-main.175.pdf

[^1_25]: https://liusiyi641.github.io/files/Automatic_Hallucination_Detection_for_Long_Context_Documents.pdf

[^1_26]: https://dataloop.ai/library/model/tasksource_deberta-small-long-nli/

[^1_27]: https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/

[^1_28]: https://aclanthology.org/2024.emnlp-main.837/

[^1_29]: https://www.reddit.com/r/MachineLearning/comments/1i0g71d/project_hallucination_detection_benchmarks/

[^1_30]: https://aclanthology.org/2024.emnlp-industry.113/

[^1_31]: https://www.ijcai.org/proceedings/2024/0687.pdf

[^1_32]: https://arxiv.org/html/2510.00880v1

[^1_33]: https://github.com/EdinburghNLP/awesome-hallucination-detection

[^1_34]: https://ashpress.org/index.php/jcts/article/download/209/164

[^1_35]: https://aclanthology.org/2025.semeval-1.123.pdf

[^1_36]: https://arxiv.org/html/2509.21357v1

[^1_37]: https://arxiv.org/html/2508.08285v2

[^1_38]: https://arxiv.org/html/2404.01210v2

[^1_39]: https://huggingface.co/vectara/hallucination_evaluation_model

[^1_40]: https://openreview.net/forum?id=YFOg1LUGG1

[^1_41]: https://developer.nvidia.com/blog/addressing-hallucinations-in-speech-synthesis-llms-with-the-nvidia-nemo-t5-tts-model/

[^1_42]: https://aclanthology.org/2023.acl-long.634.pdf

[^1_43]: https://arxiv.org/pdf/2406.00975.pdf

[^1_44]: https://zilliz.com/ai-faq/what-techniques-can-be-used-to-detect-hallucinations-in-a-raggenerated-answer-for-example-checking-if-all-factual-claims-have-support-in-the-retrieved-text

[^1_45]: https://arxiv.org/html/2404.01210v1

[^1_46]: https://arxiv.org/abs/2401.00396

[^1_47]: https://www.nature.com/articles/s41586-024-07421-0

[^1_48]: https://github.com/ParticleMedia/RAGTruth

[^1_49]: https://openreview.net/pdf/ed1130d1fa572850928fd91edeaa24e56262606b.pdf

[^1_50]: https://arxiv.org/html/2407.04121v1

[^1_51]: https://arxiv.org/html/2503.10702v1

[^1_52]: https://arxiv.org/html/2503.15354v1

[^1_53]: https://arxiv.org/html/2510.19310v1

[^1_54]: https://arxiv.org/html/2412.15189v3

[^1_55]: https://arxiv.org/html/2403.11903v1

[^1_56]: https://aclanthology.org/2025.findings-acl.765.pdf

[^1_57]: https://arxiv.org/html/2411.02400v1

[^1_58]: https://cogcomp.seas.upenn.edu/papers/Chen24.pdf

