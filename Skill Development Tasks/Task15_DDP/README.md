# Task 15: Distributed Data Parallel (DDP) Multi-Process Training using PyTorch

## Objective

This project demonstrates Distributed Data Parallel (DDP) training using PyTorch. The objective is to understand how multiple processes can collaboratively train a neural network by synchronizing gradients and updating shared model parameters efficiently.

This implementation is designed for a single-machine environment using the **GLOO** backend, making it suitable for systems without multiple GPUs.

---

## Features

- Distributed training using PyTorch DDP
- Multi-process execution
- Automatic gradient synchronization
- Distributed dataset partitioning using `DistributedSampler`
- Feedforward neural network for binary classification
- Cross-Entropy loss with Adam optimizer
- Single-machine implementation using the GLOO backend

---

## Technologies Used

- Python 3.10+
- PyTorch
- Torch Distributed
- NumPy

---

## Project Structure

```
Task15_DDP/
│
├── train_ddp.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Program

```bash
python train_ddp.py
```

---

## Workflow

1. Initialize distributed process group.
2. Spawn multiple training processes.
3. Partition the dataset using `DistributedSampler`.
4. Wrap the model with `DistributedDataParallel`.
5. Perform forward propagation.
6. Compute Cross-Entropy loss.
7. Execute backward propagation.
8. Synchronize gradients across processes.
9. Update model parameters using Adam.
10. Repeat for multiple epochs.

---

## Expected Output

```
Starting Process 0
Starting Process 1

Rank 0 | Epoch 1/5 | Loss = ...
Rank 1 | Epoch 1/5 | Loss = ...

...

Training Complete
```

---

## Learning Outcomes

After completing this project, you will understand:

- Distributed Data Parallel (DDP)
- Multi-process model training
- Gradient synchronization
- Distributed data loading
- Process group initialization
- Parallel deep learning workflows

---

## Limitations

This implementation uses the **GLOO backend** and executes on a single machine. The original task specification mentions Docker Compose, NCCL, and multi-node distributed training, which require multiple GPUs or multiple machines and are beyond the capabilities of a standard local setup.

---

## Future Enhancements

- Multi-GPU training using NCCL
- Docker Compose deployment
- Multi-node distributed execution
- Mixed Precision Training (AMP)
- Distributed checkpointing
- Performance benchmarking and scalability analysis