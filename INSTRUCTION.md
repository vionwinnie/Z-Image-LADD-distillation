# **Take-Home Assignment: LADD Distillation of Z-Image**

## **Overview**

You'll implement **LADD (Latent Adversarial Diffusion Distillation)** on the **Z-Image model**, owning the entire pipeline end-to-end — from sourcing and cleaning data, to integrating the distillation architecture, to running a full 20,000-step training run on an 8-GPU node.

The goal is a working distilled **student model** that can run inference, along with a clean, well-understood codebase you can walk through in detail.

You may use any AI coding tools, but you are expected to understand every line of code in your submission and be able to explain all design choices.

## **References**

* **LADD paper:** [https://arxiv.org/abs/2403.12015](https://arxiv.org/abs/2403.12015)  
* Z-image repo: https://github.com/Tongyi-MAI/Z-Image  
* **Z-Image training entrypoint:** https://github.com/aigc-apps/VideoX-Fun/blob/6a95acc2b71b162d903104e2d1fc36936ec87fd0/scripts/z\_image/train.py  
* You may adapt any existing open-source repos, but document what you adapted and why.  
* Feel free to resource any other relevant repos that could be helpful in your project.

## **Compute & Infrastructure**

You will be given access to a **runpod** environment with the following setup:

* **100 GB persistent storage** for your code, data, and checkpoints. Create the persistent storage in the region where 8xA100 nodes are available with name candidate1.  Familiarize yourself with the runpod setup by reading docs or watching YouTube videos.  
* **CPU instance or single-GPU instance** for development, debugging, and smoke testing. Use this for all coding and iteration work.   
* You can keep one CPU instance working for the whole week, but GPU instances must be shut down when you stop working.  
* **One node of 8× A100 GPUs** for the final training run **only**.  
  * ⚠️ **The 8-GPU node must remain inactive until you are ready to launch final training.** Do not use it for development or testing.

You can email or schedule a meeting with the hiring manager at any time for clarifying questions or to discuss your approach.

**Due date: Next Monday (one week from receipt of this assignment).**

## **Deliverables**

### **1\. Data Pipeline**

Source and prepare a dataset suitable for LADD distillation of Z-Image.

**Requirements:**

* The dataset must satisfy the base input format expected by the Z-Image training script (images, latent targets, and any conditioning signals).  
* Include a **tiny debug split** (50–200 samples) for fast local iteration.  
* Document your data sourcing decisions: what dataset(s) you used, any filtering or quality criteria applied, and how you cleaned the data.

**Output:**

* Dataset folder with a clear directory structure.  
* A short `README` covering: the data format, how the dataset was created/sourced, how to point the training script at it, and any preprocessing scripts used.

### **2\. LADD Architecture Integration**

Integrate LADD distillation into the Z-Image training pipeline.

**Requirements:**

* Implement the **teacher** and **student** model setup as described in the LADD paper.  
* The integration should be clean and togglable — it must be easy to switch between:  
  * Baseline Z-Image training  
  * Z-Image \+ LADD distillation (e.g., via a `--use_ladd` flag or config option)  
* Implement the discriminator component required by LADD, integrated correctly into the training loop.

**Output:**

* Code changes that add the LADD distillation option to the training pipeline.  
* A brief architectural note (can be in a README or inline comments) explaining: what you changed, where losses are applied, and any tradeoffs you made.

### **3\. Local Debugging & Smoke Testing**

Before touching the 8-GPU node, make the pipeline robust locally on your CPU/single-GPU instance.

**Requirements:**

* Run a single forward pass with a tiny batch; verify tensor shapes, dtypes, and device placement.  
* Confirm that gradients flow correctly through both the student model and the discriminator.  
* Add basic logging: loss values (student loss, adversarial loss), gradient norms, and memory usage if feasible.  
* Verify checkpoint save and load works end-to-end.

**Output:**

* A smoke test script (e.g., `smoke_test.py`) that can be run on a single GPU or CPU to validate the full forward/backward pass.  
* A short **debug checklist** noting what you verified and any issues you encountered and resolved.

### **4\. Full Training Run on 8-GPU Node**

Once smoke tests pass, launch the full distillation training run.

**Requirements:**

* Run on all 8 A100 GPUs using the repo's distributed training setup (DDP / `torchrun` / `accelerate` — whichever the repo uses).  
* Train for **20,000 steps**.  
* Log loss curves and save checkpoints at regular intervals.

**Output:**

* The exact launch command used.  
* Training logs (loss curves, any relevant metrics).  
* Final checkpoint path(s) on the shared storage.

### **5\. Inference Demo**

Run inference with the distilled student model and present results.

**Requirements:**

* Generate sample outputs from the student model using at least a few representative prompts.  
* Compare qualitatively (side-by-side if possible) against the teacher model or the baseline Z-Image outputs.  
* Note inference speed (steps-to-image, latency) relative to the teacher — this is a core motivation of distillation.

**Output:**

* Inference script (e.g., `inference.py`) with clear usage instructions.  
* Sample generated images with the prompts used.  
* A brief commentary on output quality and speed.

## **Presentation**

You will walk the hiring manager through your work covering:

1. **Data pipeline** — what data you used, how you cleaned it, and why.  
2. **Architecture** — how LADD is integrated into Z-Image, with a walkthrough of the relevant code.  
3. **Training** — your distributed training setup, the launch command, and the loss curves.  
4. **Inference** — live or pre-run demo of the student model.  
5. **Reflection** — what worked, what didn't, and what you'd do differently with more time.

Be prepared to answer detailed questions about any part of the codebase, including code written with AI assistance.

## **What "Success" Looks Like**

By the end, we should be able to:

* Point the training script at your prepared dataset.  
* Enable LADD distillation via a flag or config.  
* Launch distributed training across 8 GPUs and observe stable, decreasing loss.  
* Run inference from the distilled student model and see coherent image outputs.  
* Hear you clearly explain every component of the pipeline.

