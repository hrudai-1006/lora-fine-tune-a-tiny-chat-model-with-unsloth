# LoRA Fine-Tune a Tiny Chat Model with Unsloth

Build an end-to-end LoRA fine-tuning pipeline for a 4-bit Qwen2.5-0.5B chat model using Unsloth. You'll load the quantized base, attach LoRA adapters, format a tiny instruction dataset, run a featherweight SFT job, and generate a reply from the tuned model.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_base_model_and_tokenizer
- [x] **2.** count_total_parameters
- [x] **3.** is_model_4bit_quantized
- [x] **4.** ensure_pad_token
- [x] **5.** get_lora_target_modules
- [x] **6.** attach_lora_adapters
- [x] **7.** count_trainable_parameters
- [x] **8.** trainable_fraction
- [x] **9.** build_instruction_examples
- [x] **10.** format_instruction_example
- [x] **11.** format_all_examples
- [x] **12.** build_text_dataset
- [x] **13.** tokenize_text
- [x] **14.** count_tokens
- [x] **15.** build_training_arguments
- [ ] **16.** build_sft_trainer
- [ ] **17.** run_sft_training
- [ ] **18.** switch_to_inference_mode
- [ ] **19.** build_chat_prompt
- [ ] **20.** generate_reply

---

Built on Deep-ML.
