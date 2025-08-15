# 🌍 3D CNN for Hyperspectral Image Classification  

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Overview  
This project implements a **4-layer 3D Convolutional Neural Network (~2.3M parameters)** for **hyperspectral image classification** using **spatial–spectral fusion**.  
It integrates **PCA-based spectral reduction**, **25×25 window sampling**, and **Adam optimizer tuning** to deliver **state-of-the-art classification accuracy** on benchmark datasets.  

---

## 📊 Results Summary  

| Dataset       | Overall Accuracy | Kappa  | Classes | Window Size |  
|---------------|------------------|--------|---------|-------------|  
| Indian Pines  | **99%+**         | 0.99   | 16      | 25×25        |  
| Salinas       | **99%+**         | 0.99   | 16      | 25×25        |  
| Pavia Univ.   | **98%+**         | 0.98   | 9       | 25×25        |  

---

## 🏗 Model Architecture  

```
Input (25×25×Bands)  
│  
├── Conv3D (ReLU) ×4  
├── BatchNormalization  
├── Dropout (Regularization)  
├── Flatten  
├── Dense (Fully Connected) ×2  
└── Softmax Output (Classes)  
```

---

## 🚀 Key Features  
- **High Accuracy:** Achieved **>99% OA** and **0.99 Kappa** on Indian Pines dataset.  
- **Efficient Processing:** PCA + window sampling reduced computation by ~70% without accuracy drop.  
- **Robust Evaluation:** Outputs OA, AA, per-class accuracy, Kappa, and confusion matrices.  
- **Reusability:** Modular pipeline adaptable to multiple hyperspectral datasets.  
- **Scalable:** Easy to extend to different image sizes and spectral resolutions.  

---

## 📂 Dataset Links  
- [Indian Pines Dataset](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)  
- [Salinas Dataset](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)  
- [Pavia University Dataset](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)  

---

## ⚙️ Installation  

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-cnn-hsi.git
cd 3d-cnn-hsi

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Training  

```python
python train.py --dataset IP --windowSize 25 --pcaComponents 30
```

---

## 📜 License  
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements  
This work is inspired by research on **hyperspectral image classification** and utilizes open-source datasets for reproducibility.
