# 🤖 LLM Engineering @ UTHU

Welcome to the **LLM Engineering Track** — your deep dive into building, tuning, evaluating, and deploying Large Language Models (LLMs). From Transformers to fine-tuning, from RAG to evaluation pipelines — it's all here.

Built with ⚡ **Colab-friendly labs**, **real projects**, and **research-ready design patterns**.

---

## 🧭 Track Structure

The track is organized into 6 sub-books:

1. [LLM Fundamentals](#1-llm-fundamentals)
2. [Pretraining & Finetuning](#2-pretraining--finetuning)
3. [RAG Systems](#3-rag-systems)
4. [LLM Deployment](#4-llm-deployment)
5. [LLM Evaluation](#5-llm-evaluation)
6. [Advanced Topics](#6-advanced-topics)

---

## 📘 1. LLM Fundamentals

> Transformer math, tokenizer tricks, prompt patterns.

| Notebook | Description |
|----------|-------------|
| `01_transformer_architecture_in_depth.ipynb` | QKV, attention scores, multi-heads |
| `02_tokenization_bytepair_unicode.ipynb` | BPE, WordPiece, SentencePiece |
| `03_prompt_engineering_patterns.ipynb` | Zero-shot, CoT, few-shot formats |
| `04_scaling_laws_and_compute_optimization.ipynb` | Chinchilla vs GPT3 training cost curves |
| `05_model_architectures_gpt_llama_mistral.ipynb` | GPT2 → LLaMA → Mistral architecture tour |
| `06_attention_optimizations_flash_paged.ipynb` | FlashAttention, xFormers, paged attention |
| `07_lab_tokenizer_visualizer_and_custom_vocab.ipynb` | BPE/SentencePiece tokenizer visualization |
| `08_lab_transformer_forward_pass_step_by_step.ipynb` | Implement one transformer block manually |
| `09_lab_prompt_patterns_and_token_logprobs.ipynb` | Prompt → logits → logprobs exploration |

---

## 📘 2. Pretraining & Finetuning

> Training from scratch and adapting to new tasks.

| Notebook | Description |
|----------|-------------|
| `01_data_pipelines_for_pretraining.ipynb` | Load, clean, tokenize large datasets |
| `02_distributed_pretraining_with_megatron.ipynb` | Use Megatron-LM for multi-GPU pretraining |
| `03_parameter_efficient_finetuning.ipynb` | LoRA, QLoRA, adapters |
| `04_instruction_finetuning_alpaca_format.ipynb` | Finetune on instruct datasets |
| `05_rlhf_reward_modeling_ppo.ipynb` | Reward model + PPO loop |
| `06_domain_adaptation_medical_legal_finetuning.ipynb` | Finetune for healthcare/legal data |
| `07_lab_tiny_gpt2_pretraining_from_scratch.ipynb` | GPT-2 tiny pretrain on custom data |
| `08_lab_parameter_efficient_finetune_lora.ipynb` | Apply LoRA to large model |
| `09_lab_rlhf_reward_model_mock_demo.ipynb` | Simulate RLHF pipeline locally |

---

## 📘 3. RAG Systems

> Combine retrieval with generation for grounded, contextual answers.

| Notebook | Description |
|----------|-------------|
| `01_vector_databases_pinecone_weaviate.ipynb` | Use vector DBs for LLM context |
| `02_advanced_retrieval_hybrid_search.ipynb` | BM25 + dense fusion search |
| `03_document_chunking_and_metadata.ipynb` | Sliding windows, overlapping chunks |
| `04_evaluation_with_ragas_trl.ipynb` | RAG eval metrics: faithfulness, grounding |
| `05_multimodal_rag_images_tables.ipynb` | RAG with image embeddings |
| `06_production_rag_with_llamaindex.ipynb` | RAG in production: pipelines |
| `07_lab_chunking_and_embedding_evaluation.ipynb` | Compare chunking strategies |
| `08_lab_vector_search_pipeline_with_chroma.ipynb` | Build end-to-end RAG with ChromaDB |
| `09_lab_metadata_filtering_in_retrieval.ipynb` | Tag-based hybrid search filter demo |

---

## 📘 4. LLM Deployment

> From colab to cluster. Serve your models.

| Notebook | Description |
|----------|-------------|
| `01_serving_frameworks_vllm_tgi.ipynb` | Latency tradeoffs of vLLM vs TGI |
| `02_quantization_ggml_awq_gptq.ipynb` | Compress large models to 4-bit |
| `03_distributed_inference_tensorrt_llm.ipynb` | Run models at scale with TensorRT-LLM |
| `04_edge_deployment_ollama_mlc.ipynb` | Deploy to iPhones and edge with MLC |
| `05_caching_and_request_batching.ipynb` | Speedup via KV cache + batch queue |
| `06_cost_monitoring_and_autoscaling.ipynb` | Inference cost vs throughput dashboards |
| `07_lab_vllm_vs_tgi_latency_comparison.ipynb` | Run latency benchmark suite |
| `08_lab_quantize_with_gptq_and_awq.ipynb` | Quantize Mistral using GPTQ |
| `09_lab_batching_and_request_queuing_testbed.ipynb` | Test concurrency and load queue |

---

## 📘 5. LLM Evaluation

> Not all answers are good answers — test robustness, fairness, truth.

| Notebook | Description |
|----------|-------------|
| `01_automated_metrics_bleu_rouge_bertscore.ipynb` | Metric theory and math |
| `02_human_eval_setups_and_crowdsourcing.ipynb` | Setup scalable annotator workflows |
| `03_toxicity_bias_detection.ipynb` | Prompt probing, fairness testing |
| `04_red_teaming_adversarial_testing.ipynb` | Jailbreak prompts, prompt injection |
| `05_latency_throughput_benchmarking.ipynb` | Real-time performance profiling |
| `06_model_cards_and_audit_trails.ipynb` | Build audit logs and datasheets |
| `07_lab_bleu_rouge_bertscore_eval_suite.ipynb` | Run all major NLP metrics end-to-end |
| `08_lab_human_eval_grading_interface.ipynb` | Build human eval UX with Gradio |
| `09_lab_bias_and_toxicity_metrics_demo.ipynb` | TOX/BIAS detectors in practice |
| `10_lab_red_teaming_simulation.ipynb` | Adversarial prompts vs defenses |
| `11_lab_latency_benchmarking_with_vllm_vs_ggml.ipynb` | Compare vLLM/ggml latency/perf |
| `12_lab_model_card_generator_pipeline.ipynb` | Generate model cards automatically |

---

## 📘 6. Advanced Topics

> Next-gen systems for frontier AI engineers.

| Notebook | Description |
|----------|-------------|
| `01_mixture_of_experts_implementation.ipynb` | Load-balance with expert routing |
| `02_long_context_processing_ring_attention.ipynb` | 32k+ context + efficient attention |
| `03_multi_agent_llm_systems.ipynb` | Chat between agents via tools/memory |
| `04_llm_os_agi_prototyping.ipynb` | Simulate autonomous agent tasks |
| `05_compression_sparse_pruning.ipynb` | LoRA + Pruning + Sparse fusion |
| `06_energy_efficient_llms.ipynb` | Carbon costs, low-power tuning |
| `07_lab_moe_switch_transformer_inference.ipynb` | Demo expert gating behavior |
| `08_lab_long_context_test_rag_vs_ringattention.ipynb` | Compare RAG vs Ring Attention |
| `09_lab_multi_agent_llm_scratchpad_protocol.ipynb` | Reasoning memory + agent tools |

---

## 🧠 Prereqs

- Transformers 101 knowledge
- Python + NumPy + PyTorch
- Hugging Face basics
- GPU (Colab Pro works)

---

## 🔥 You’ll Be Able To…

- Train/Finetune your own LLMs  
- Deploy them at scale  
- Evaluate safety, latency, and hallucination  
- Build multi-agent + RAG + quantized stacks  
- Teach others how to do it

---

## 🚀 Pro Tip

> Pair this with `06_mlops` to move toward full-stack LLMOps.

---

