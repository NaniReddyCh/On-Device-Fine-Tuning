# Samsung EnnovateX 2025 AI Challenge Submission

- **Problem Statement:** On-Device Fine-Tuning Framework for Billion+ Parameter Scale LLMs
- **Team Name:** ByteBots
- **Team Members:** Srikrishna, Srikanth, Kiran, Vishweshwar
<!-- - **Demo Video Link:** *(Upload the Demo video on Youtube as a public or unlisted video and share the link. Google Drive uploads or any other uploads are not allowed.)* -->

---

## Project Artefacts

- **Technical Documentation:** [docs/](docs)  
  All technical details are provided in markdown files inside the `docs` folder.
- **Source Code:** [src/](src)  
  All source code is in the `src` folder. The code is installable and runs consistently on intended platforms.
- **Models Used:** [`gpt2`](https://huggingface.co/gpt2), [`TinyLlama`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-tokenizer)  
  All models are sourced from Hugging Face and used for experimentation and benchmarking.
- **Models Published:**  
  _(If any models are developed and published, add Hugging Face links here.)_
- **Datasets Used:**
  - [SST-2 (Stanford Sentiment Treebank v2)](https://huggingface.co/datasets/glue/viewer/sst2) &mdash; Used for sentiment classification and fine-tuning experiments. Licensed for research use via Hugging
- **Datasets Published:**  
  _(List and link any datasets created and published for this project.)_

---

## Attribution

Please refer to the detailed documentation in the [docs/](docs) folder.

---

# PRGE: On-Device LLM Fine-Tuning with Parallelized Randomized Gradient Estimation

This repository contains research code for **on-device, privacy-preserving fine-tuning** of large language models (LLMs) using LoRA adapters and PRGE (Parallelized Randomized Gradient Estimation), a zeroth-order (gradient-free) optimization technique.  
It is designed for efficient, parameter-efficient adaptation on user data, suitable for edge devices.

## Features

- **PRGE (Parallelized Randomized Gradient Estimation):** Efficient, gradient-free, forward-only optimization for fine-tuning LLMs using only inference engines.
- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning by updating only small adapter modules.
- **On-Device Focus:** All computation and data remain on the user's device for privacy and security.
- **Streamlit UI:** Simple web interface for running and visualizing fine-tuning.
- **Model Saving:** Fine-tuned models and tokenizers are saved for later use.
- **Mobile Export:** `mobile_export.py` demonstrates exporting the merged (base + LoRA) model to TorchScript or ExecuTorch format for Android or other mobile runtimes.

## Directory Structure

```
On-Device-Fine-Tuning/
├── main.py           # Streamlit UI entry point
├── train.py          # Training loop and evaluation
├── prge_optimizer.py # PRGE zeroth-order optimizer implementation
├── model_utils.py    # Model preparation and LoRA utilities
├── dataset_utils.py  # Data loading and preprocessing
├── test.py           # Script to test the fine-tuned model
├── plots.py          # Plotting utilities
├── mobile_export.py  # Export fine-tuned model for mobile/edge runtimes
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**

   ```sh
   streamlit run main.py
   ```

3. **Fine-tune a model:**

   - Select model and hyperparameters in the UI.
   - Click "Start Training".
   - Fine-tuned models are saved in timestamped folders.

4. **Test your model:**
   - Use `test.py` to load and generate text with your fine-tuned model.

## Example: Testing a Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "fine_tuned_model_YYYYMMDD_HHMMSS"  # replace with your folder
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Your test prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- For gated models (like Gemma), you must have access and be authenticated with Hugging Face.
- Results are saved to `results.json` during training.
- Only LoRA adapter weights are updated for efficiency.

## More Details

For a comprehensive explanation of the methodology, experiments, and results, please refer to the attached paper: [PRGE_fine_tuning.pdf](docs/PRGE_fine_tuning.pdf) included in this repository.

## References

- **Research Paper:** [Enabling Efficient On-Device Fine-Tuning of LLMs Using Only Inference Engines](https://arxiv.org/abs/2409.15520)
- **Medium Article:** [Fine-tuning Gemma with LoRA for On-Device Inference (Android, iOS, Web) with Separate LoRA Weights](https://medium.com/@denisov.shureg/fine-tuning-gemma-with-lora-for-on-device-inference-android-ios-web-with-separate-lora-weights-f05d1db30d86)

## License

This code is for research and educational purposes only.  
Built as part of the **Samsung EnnovateX Hackathon 2025**.

---

## Authors

**Team ByteBots**  
Srikrishna, Srikanth, Kiran, Vishweshwar  
Students from IIT Hyderabad
