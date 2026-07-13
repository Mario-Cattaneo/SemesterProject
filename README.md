# Adaptive Activation Compression & Early-Exit for DAG-Split Collaborative Inference

## 1. Introduction & Motivation

The rapid proliferation of Internet of Things (IoT) and edge devices has driven the demand for real-time, on-device deep learning inference. However, executing modern, large-scale Deep Neural Networks (DNNs) on individual resource-constrained edge nodes is often infeasible due to strict memory, computational, and energy limitations. 

To address this bottleneck, **collaborative edge inference** has emerged as a promising paradigm. By partitioning a DNN vertically (layer-wise) and horizontally (spatial-wise), computation can be distributed across a Directed Acyclic Graph (DAG) of cooperative edge devices and local servers. 

While collaborative splitting reduces the local computational burden, it introduces a new challenge: the latency overhead of transmitting intermediate activation tensors over bandwidth-limited wireless channels. If these activations are transmitted in their raw, uncompressed format, the communication delay can easily exceed the computational savings of distribution.

This project addresses this challenge by researching, developing, and analyzing **task-oriented activation compression** and **early-exit mechanisms** designed specifically for arbitrary DAG-split DNNs. Unlike traditional data-oriented compression, task-oriented compression discards task-irrelevant features, retaining only the semantic information necessary for the downstream inference task, thereby achieving extreme compression ratios and enabling low-latency edge intelligence.

---

## 2. Project Objectives & Design Goals

The primary objective of this project is to research, compare, develop, and analyze the accuracy-to-latency trade-off of various compression and execution approaches on a DAG-split DNN. 

To achieve this, we are building an extensible, modular research framework with the following design goals:

### A. Extensible Research Interface
We require a unified, clean interface to easily add, modify, train, and analyze new compression and coding approaches. This interface must allow us to evaluate different families of compression (e.g., uniform quantization, magnitude pruning, and learning-based bottlenecks) under identical system constraints.

### B. Comprehensive System Modeling
The framework must allow us to explicitly specify and simulate:
*   **Available Devices:** The number of physical nodes ($K$) in the mesh network.
*   **Device Capabilities:** Heterogeneous hardware profiles, including local computation speeds (processing coefficients) and local power states.
*   **Channel Conditions:** The transmission bandwidth (Mbps) of the wireless links, with the ability to toggle between noiseless digital channels and noisy analog channels.
*   **Splitting Optimization Parameters:** The parameters used by the underlying splitting engine (such as DiSNet) to decide the optimal vertical and horizontal partition ratios.

### C. Early-Exit Decision Making
We aim to integrate an **early-exit mechanism** into the collaborative DAG. Each device on the active inference path should be capable of hosting a lightweight exit classifier. If the prediction confidence (e.g., softmax entropy) at an intermediate stage exceeds a specified threshold, the system exits early, outputting the prediction locally and completely bypassing the remaining transmission and computation stages of the DAG.

---

## 3. Current Standpoint & Technical Findings (What We Know)

Our research is currently built on top of the **DiSNet** splitting framework, using a pre-trained VGG-16 backbone. Through a deep-dive analysis of the DiSNet codebase, we have uncovered several critical mechanics of how it represents and executes splits:

*   **Spatial Slicing (Dimension 2):** DiSNet performs horizontal partitioning by **slicing the Height dimension (dimension 2)** of the feature maps, not the channels. For example, a $112 \times 112$ feature map is sliced into horizontal strips and distributed to parallel devices.
*   **Receptive Field Overlap & Trimming:** Because convolutions have a kernel size (e.g., $3 \times 3$), local spatial slices must overlap with their neighbors to compute edge pixels correctly. DiSNet dynamically calculates these overlaps, runs the local convolutions, and then trims the boundary pixels before merging.
*   **Immediate Layer-by-Layer Merging:** Inside `opt_DiSNet`, the spatial slices are **concatenated back together along the Height dimension at the end of every single layer**. The model does not remain split across multiple hops; it splits, computes one layer, merges, and splits again for the next layer.
*   **VGG-16 Vertical Stages:** The 18 layers of VGG-16 are grouped into 4 vertical stages using `layer_range = np.array([[0, 4], [4, 9], [9, 14], [14, 18]])`.

---

## 4. Critical Issues Diagnosed & Resolved (The Technical Hurdles)

During our initial integration and simulation runs in Google Colab, we diagnosed and resolved several critical environment and architectural conflicts:

*   **The Python 2 `super()` Reload Bug:** The original model file (`models/model_vgg16.py`) used `super(VGG16, self).__init__()`. In Jupyter/Colab, reloading modules causes the class registry to duplicate, making this call fail with a `TypeError`. We patched this to use the modern, robust Python 3 `super().__init__()`.
*   **The `torch.hub` Version Mismatch:** The original `main/model_convert.py` requested `torchvision v0.9.0` via `torch.hub`, which is incompatible with modern PyTorch in Colab, causing an ONNX import error. We patched the script to load VGG-16 directly from `torchvision.models`.
*   **The Hardcoded Paths:** `main/model_convert.py` was hardcoded to look for the model checkpoint in the author's local directory (`/home/eric/`). We patched the path to point to Colab's root cache directory (`/root/.cache/`).
*   **The Gradient Tracking Conflict:** `opt_DiSNet` is an inference-only engine filled with non-differentiable slicing and list manipulations. Passing a tensor with active gradients (`requires_grad=True`) through it during training caused its internal spatial calculators to fail silently, leading to invalid spatial shapes (height of 1) and triggering a MaxPool `RuntimeError`. We resolved this by implementing **Offline Activation Training**, where activations are collected safely under `no_grad` and the bottlenecks are trained offline.
*   **The Batch Size Constraint:** DiSNet's spatial partition calculator is strictly hardcoded for a batch size of 1. Training with a batch size greater than 1 causes silent shape mismatches. We aligned our training loop to use a batch size of 1.

---

## 5. Target Architecture & Implementation Blueprint (What We Want)

To avoid fighting DiSNet's fragile internal execution loop, we are transitioning to a **decoupled architecture** that separates the **System Model** (the graph and partition calculations) from the **Execution Model** (the PyTorch forward pass).

### A. Decoupled Execution Model (`DAGEngine`)
We want to implement an explicit execution engine (`DAGEngine`) that:
1.  Extracts the VGG-16 layers into clean, vertical stages based on `configurations.layer_range`.
2.  Performs spatial slicing along the Height dimension using the exact indices calculated by DiSNet's `FieldCalculatorDiSNet`.
3.  Applies our activation compression schemes directly to the individual spatial slices before they are processed.
4.  Merges the spatial slices back together at the end of each layer.

### B. Unified Coding Interface (`BaseCodingScheme`)
We want a unified interface (`BaseCodingScheme`) that supports both standard and task-oriented compression schemes. The interface must support:
*   **Dynamic Adaptation:** During evaluation, the scheme must accept the `bandwidth` (Mbps) and `target_latency` (seconds) and dynamically calculate its optimal compression parameters (e.g., bit-width, pruning ratio, or spatial downsampling factor) to satisfy the latency constraint.
    *   *Latency Constraint Formula:* $\text{Latency}_{\text{comm}} = \frac{\text{Size} \times \text{bits}}{\text{Bandwidth}} \le \text{Target Latency}$
*   **Offline Training Phase:** A training method (`train_scheme`) to fine-tune bottleneck layers. For learning-based schemes (like VDDIB), we will use **Offline Activation Training** to avoid gradient conflicts with DiSNet. We run the simulation forward pass under `no_grad` to collect the original activations, and then train the bottleneck layers offline to minimize the Reconstruction Loss (MSE) between the original and reconstructed activations.

### C. The Compression Schemes to Implement
We want to implement and compare three distinct coding families:
1.  **Uniform Quantization:** Dynamically calculates the maximum allowed bit-width $n$ (capped between 2 and 32 bits) to satisfy the latency constraint, and quantizes the activations uniformly.
2.  **Magnitude Pruning:** Dynamically calculates the maximum allowed keep-ratio (capped between 0.05 and 1.0) to satisfy the latency constraint, and prunes the smallest activations by magnitude.
3.  **DAG-VDDIB (Proposed):** A trainable, dimension-aware bottleneck. It dynamically searches for the optimal combination of spatial downsampling factor $F$ and quantization bits $n$ to satisfy the latency constraint:
    $$\text{Effective Bits} = \frac{n}{F^2} \le \frac{\text{Target Latency} \times \text{Bandwidth}}{\text{Size}}$$
    It then uses a lightweight encoder/decoder network (using `nn.Conv2d` for 4D feature maps and `nn.Linear` for 2D dense vectors) trained offline to reconstruct the activations.

### D. The Evaluation Requirements
We want to evaluate the schemes under a **constant network bandwidth** (e.g., 5.0 Mbps) across a range of **target latencies** (e.g., 0.05s, 0.1s, 0.2s, 0.5s, 1.0s). For each target latency:
1.  The coding scheme must adapt its parameters to the constraint.
2.  If the scheme is trainable (like VDDIB), it must train its bottleneck layers offline.
3.  The engine must execute the collaborative inference and measure the final Top-5 accuracy.
4.  We plot the resulting **Accuracy vs. Target Latency** curves for all three schemes on a single graph to compare their performance.

---

## 6. Instructions for Gemini in Colab

When you open your Colab notebook and activate Gemini, paste the following prompt to get started:

> "Hello Gemini. I am working on a collaborative edge inference project using the DiSNet repository. I have attached a `README.md` file that details our project objectives, our architectural findings, the environment patches we applied, and our target decoupled architecture.
>
> Please read the `README.md` carefully. Your first task is to inspect the files in `/content/DiSNet/` (using python or bash commands) to locate exactly where `create_partition_input` is defined, update our import statements accordingly, and then help me implement the decoupled `DAGEngine` and the coding schemes described in the document."
