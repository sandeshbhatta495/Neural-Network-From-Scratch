# Neural Networks From Scratch - Learning Journey

> A comprehensive exploration of Neural Networks based on **"Neural Networks from Scratch in Python"** by Harrison Kinsley and Daniel Kukiela

---

## 📚 Project Overview

This repository documents my learning journey through the NNFS book, implementing neural networks from first principles without relying solely on high-level frameworks. Each chapter builds upon foundational concepts to develop a complete understanding of how neural networks work.

**Book Reference**: Neural Networks from Scratch in Python (PDF included in project root)

---

## ✅ Completed Chapters

### **Chapter 01: Fundamentals**
- **Status**: ✅ Completed
- **Key Concepts**:
  - Introduction to Neural Networks
  - Basic mathematical foundations
  - Dot products and their role in neural computations
  - Preparing for layer operations
- **Notebooks**:
  - `dot product.ipynb` - Understanding dot product operations
  - `for a layer.ipynb` - Introduction to layer processing

---

### **Chapter 02: Single and Multi-Neuron**
- **Status**: ✅ Completed
- **Key Concepts**:
  - Single neuron architecture and operations
  - Multi-neuron networks
  - Matrix operations and their efficiency
  - Numpy integration for numerical computing
  - Manual calculation without Numpy for understanding
- **Notebooks**:
  - `single neuron.ipynb` - Single neuron implementation
  - `single neuron calculation without numpy.ipynb` - Manual calculations
  - `matrix.ipynb` - Matrix operations and batching
  - `using loops.ipynb` - Iterative approaches to network computation

---

### **Chapter 03: Hidden Layers**
- **Status**: ✅ Completed
- **Key Concepts**:
  - Hidden layer architecture
  - Dense/Fully-Connected layers
  - NNFS library fundamentals
  - Basic layer implementation and integration
- **Notebooks**:
  - `Dense Layer.ipynb` - Implementation of dense layers
  - `Dense Layer Basics.ipynb` - Foundational concepts
  - `hidden layer 1.ipynb` - First hidden layer experiments
  - `nnfs library.ipynb` - NNFS custom library usage

---

### **Chapter 04: Activation Functions**
- **Status**: 📋 In Progress (Folder Created, Content to be Added)
- **Key Concepts**:
  - Need to implement: ReLU, Sigmoid, Tanh, Softmax
  - Activation function properties and use cases
  - Non-linearity introduction
  - Expected notebooks:
    - `activation functions comparison.ipynb`
    - `relu implementation.ipynb`
    - `softmax for classification.ipynb`

---

## 📋 Upcoming Chapters (Not Started)

### **Chapter 05: Calculating Network Output**
- Forward propagation
- Batch processing
- Network output calculations
- Performance optimization

### **Chapter 06: Loss**
- Loss functions overview
- Mean Squared Error (MSE)
- Categorical Cross-Entropy
- Binary Cross-Entropy
- Loss calculation and interpretation

### **Chapter 07: Optimization**
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Batch Gradient Descent
- Learning rate effects
- Optimization strategies

### **Chapter 08: Backpropagation**
- The Backpropagation algorithm
- Chain rule application
- Gradient computation
- Jacobian and Hessian matrices
- Computational efficiency

### **Chapter 09: Regularization**
- Overfitting and underfitting
- L1 and L2 regularization
- Weight decay
- Early stopping
- Regularization strategies

### **Chapter 10: Dropout**
- Dropout concept and implementation
- Preventing overfitting
- Dropout during training vs inference
- Inverted dropout
- Best practices

### **Chapter 11: Batch Normalization**
- Batch normalization concept
- Internal covariate shift
- Implementation details
- Training vs inference behavior
- Performance improvements

### **Chapter 12: Convolutional Neural Networks**
- Convolutional layers
- Filters and kernels
- Pooling operations
- CNN architectures
- Image processing applications

### **Chapter 13: Object Detection**
- Object detection fundamentals
- YOLO and similar architectures
- Bounding box predictions
- Multi-class detection
- Real-world applications

### **Chapter 14: Recurrent Neural Networks**
- RNN architecture
- LSTM cells
- GRU cells
- Sequence processing
- Time series applications

### **Chapter 15: Custom Objects**
- Building custom layers
- Custom loss functions
- Custom metrics
- Object-oriented design
- Extensibility patterns

### **Chapter 16: Model Saving and Loading**
- Model serialization
- Checkpoint saving
- Weight persistence
- Model reconstruction
- Version control

### **Chapter 17: Transfer Learning**
- Pre-trained models
- Fine-tuning approaches
- Feature extraction
- Domain adaptation
- Practical applications

### **Chapter 18: Advanced Topics**
- Attention mechanisms
- Transformer architecture
- Generative models
- Advanced optimization
- Research frontiers

---

## 📂 Project Structure

```
Neural Language From Scratch/
├── README.md (this file)
├── Ch 01 - Fundamentals/
│   ├── dot product.ipynb
│   └── for a layer.ipynb
├── Ch 02 - Single and Multi-Neuron/
│   ├── single neuron.ipynb
│   ├── single neuron calculation without numpy.ipynb
│   ├── matrix.ipynb
│   └── using loops.ipynb
├── Ch 03 - Hidden Layers/
│   ├── Dense Layer.ipynb
│   ├── Dense Layer Basics.ipynb
│   ├── hidden layer 1.ipynb
│   └── nnfs library.ipynb
├── Ch 04 - Activation Functions/
│   └── [Content to be added]
├── Ch 05 - Calculating Network Output/
│   └── [Content to be added]
├── Ch 06 - Loss/
│   └── [Content to be added]
├── Ch 07 - Optimization/
│   └── [Content to be added]
├── Ch 08 - Backpropagation/
│   └── [Content to be added]
├── Ch 09 - Regularization/
│   └── [Content to be added]
├── Ch 10 - Dropout/
│   └── [Content to be added]
├── Ch 11 - Batch Normalization/
│   └── [Content to be added]
├── Ch 12 - Convolutional Neural Networks/
│   └── [Content to be added]
├── Ch 13 - Object Detection/
│   └── [Content to be added]
├── Ch 14 - Recurrent Neural Networks/
│   └── [Content to be added]
├── Ch 15 - Custom Objects/
│   └── [Content to be added]
├── Ch 16 - Model Saving and Loading/
│   └── [Content to be added]
├── Ch 17 - Transfer Learning/
│   └── [Content to be added]
├── Ch 18 - Advanced Topics/
│   └── [Content to be added]
├── Neural-Networks-from-Scratch-in-Python-by-Harrison-Kinsley-Daniel-Kukiela-z-lib.org_.pdf
├── .venv/ (Python virtual environment)
└── .git/ (Version control)
```

---

## 🎯 Learning Objectives

- [ ] Understand fundamental neural network concepts
- [ ] Implement neural networks from scratch
- [ ] Master forward propagation
- [ ] Master backpropagation
- [ ] Implement and understand activation functions
- [ ] Work with loss functions
- [ ] Implement optimization algorithms
- [ ] Build and train multi-layer networks
- [ ] Prevent overfitting with regularization
- [ ] Understand and implement CNNs
- [ ] Work with RNNs and LSTMs
- [ ] Apply transfer learning
- [ ] Explore advanced architectures

---

## 🛠️ Technologies & Tools

- **Python 3.x**
- **Jupyter Notebooks** - Interactive learning and experimentation
- **NumPy** - Numerical computing and matrix operations
- **Matplotlib** - Visualization and plotting
- **NNFS Library** - Custom neural network framework (developed through course)
- **Git** - Version control

---

## 📊 Progress Summary

| Chapter | Topic | Status | Notebooks | Progress |
|---------|-------|--------|-----------|----------|
| 1 | Fundamentals | ✅ Complete | 2 | 100% |
| 2 | Single & Multi-Neuron | ✅ Complete | 4 | 100% |
| 3 | Hidden Layers | ✅ Complete | 4 | 100% |
| 4 | Activation Functions | 📋 In Progress | 0/3 | 0% |
| 5 | Calculating Network Output | ⏳ Not Started | 0/3 | 0% |
| 6 | Loss Functions | ⏳ Not Started | 0/3 | 0% |
| 7 | Optimization | ⏳ Not Started | 0/4 | 0% |
| 8 | Backpropagation | ⏳ Not Started | 0/4 | 0% |
| 9 | Regularization | ⏳ Not Started | 0/3 | 0% |
| 10 | Dropout | ⏳ Not Started | 0/2 | 0% |
| 11 | Batch Normalization | ⏳ Not Started | 0/2 | 0% |
| 12 | CNN | ⏳ Not Started | 0/5 | 0% |
| 13 | Object Detection | ⏳ Not Started | 0/3 | 0% |
| 14 | RNN/LSTM | ⏳ Not Started | 0/4 | 0% |
| 15 | Custom Objects | ⏳ Not Started | 0/3 | 0% |
| 16 | Model Serialization | ⏳ Not Started | 0/2 | 0% |
| 17 | Transfer Learning | ⏳ Not Started | 0/3 | 0% |
| 18 | Advanced Topics | ⏳ Not Started | 0/4 | 0% |

**Overall Progress**: ~16% Complete (3 of 18 chapters)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip or conda
- Jupyter Notebook/JupyterLab

### Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies: `pip install numpy matplotlib jupyter`
5. Launch Jupyter: `jupyter notebook`

---

## 📖 How to Use This Repository

1. **Read the book first** - Each chapter in the book corresponds to a folder here
2. **Follow the notebooks** - Notebooks are ordered and build upon each other
3. **Experiment** - Modify code and run experiments to deepen understanding
4. **Take notes** - Add markdown cells with your own insights
5. **Track progress** - Mark chapters as complete in your study routine

---

## 🔄 Next Steps (Priority Order)

1. **Activate Functions (Ch 04)** - Critical foundation for non-linear networks
2. **Loss Functions (Ch 06)** - Essential for training networks
3. **Optimization (Ch 07)** - Learning how networks actually train
4. **Backpropagation (Ch 08)** - Core algorithm for training (THE most important)
5. **Continue sequentially** - Build systematically through remaining chapters

---

## 💡 Key Concepts Mastered So Far

✅ **Chapter 1-3 Achievements**:
- Fundamentals of dot products and matrix operations
- Single neuron forward pass calculation
- Multi-neuron networks with batching
- Introduction to hidden layers
- Dense layer implementation
- NNFS library basics

📈 **What's Coming**:
- Non-linear transformations via activation functions
- Training concepts (loss and optimization)
- The revolutionary backpropagation algorithm
- Advanced architectures (CNN, RNN)
- Real-world applications

---

## 🎓 Learning Resources

- **Primary**: Neural Networks from Scratch in Python (included PDF)
- **Supplementary**: 
  - 3Blue1Brown Neural Networks series (YouTube)
  - Stanford CS231n (Convolutional Neural Networks)
  - Andrew Ng Deep Learning Specialization

---

## 📝 Notes & Tips

### For Future Me:
1. **Don't skip chapters** - Each builds essential foundations
2. **Implement manually first** - Before using libraries
3. **Understand math deeply** - Linear algebra is crucial
4. **Experiment with hyperparameters** - Build intuition
5. **Reference the book** - When stuck, re-read explanations

### Common Pitfalls to Avoid:
- Skipping matrix operations understanding
- Not grasping backpropagation fully before moving on
- Ignoring the mathematical foundation
- Not experimenting enough with implementations
- Using libraries before understanding fundamentals

---

## 🤝 Contributing to This Project

As I progress through the book, I'll:
1. Add notebooks for each chapter
2. Update progress in this README
3. Document key learnings
4. Create summary documents
5. Maintain clean, well-commented code

---

## 📅 Study Schedule (Projected)

- **Completed**: Chapters 1-3 (3 weeks)
- **Current Focus**: Chapter 4 (Activation Functions) - 1 week
- **Next 8 weeks**: Chapters 5-11 (Core Training Concepts)
- **Following 4 weeks**: Chapters 12-14 (Advanced Architectures)
- **Final 3 weeks**: Chapters 15-18 (Advanced Topics & Refinement)

**Estimated Total**: 19 weeks / ~4.5 months

---

## 📞 Questions & Insights

As questions arise during learning, they're documented in respective chapter notebooks. Key insights and "aha moments" are recorded for future reference.

---

## ✨ Final Goal

By the end of this journey, I will have:
- ✅ Complete understanding of neural network fundamentals
- ✅ Ability to implement networks from scratch
- ✅ Mastery of modern architecture (CNN, RNN, Transformers)
- ✅ Practical skills in model training and optimization
- ✅ Knowledge to implement research papers
- ✅ Foundation for advanced machine learning

---

**Last Updated**: January 24, 2026

**Current Status**: 🔄 Active Learning - Chapter 4 (Activation Functions)

---

*"Understanding neural networks at a fundamental level is like learning to build a house from scratch. You need to understand the foundation, walls, and structure before adding decorations."* - Learning Philosophy
