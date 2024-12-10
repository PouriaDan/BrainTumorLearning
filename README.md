### Contributors:
**Pooria Daneshvar Kakhaki**

**Neda Ghohabi Esfahani**

## Introduction: 
Brain tumors are among the most aggressive diseases, with over 84,000 individuals expected to receive a primary brain tumor diagnosis in 2021 and approximately 18,600 estimated to lose their lives to malignant brain tumors (brain cancer) during the same year [8]. Magnetic Resonance Imaging (MRI) remains the most effective method for detecting brain tumors. Unlike other cancers, brain tumors often result in profound and lasting physical, cognitive, and psychological effects on patients. Therefore, early diagnosis and the development of optimal treatment plans are crucial to improving both life expectancy and quality of life for these individuals. Neural networks have demonstrated remarkable accuracy in image classification and segmentation tasks, making them highly valuable in advancing brain tumor detection and analysis.


## Motivations:
Importance and Potential Impact of the Project
- MRI is currently the gold standard for detecting brain tumors, generating large volumes of image data for analysis.
- Effective treatment relies on precise diagnosis and classification, which can significantly impact patient survival rates and quality of life.
- Radiologists manually examine these MRI images, a process prone to errors given the complexities and subtle differences in tumor characteristics.


## Dataset
- The dataset for tumor classification can be accessed from https://www.kaggle.com/code/thomasdubail/creating-brain-tumors-dataset

This dataset is designed for advanced medical research, containing MRI images across four classes: 

- Pituitary: Abnormal growth of cells in the pituitary gland
- Glioma: Tumor that originates in the brain or spinal cord from glial cells
- Meningioma: Tumor that grows in the meninges
- Normal: No tumor
  
![image](https://github.com/user-attachments/assets/599c46cb-205f-4c50-b80e-263438e9c2be)

- The dataset for tumor segmentation can be accessed from

![image](https://github.com/user-attachments/assets/ad8adcab-ecef-4dc4-89bf-b5f87c787d26)


## Method

- **Tumor Classification**: Implementing Convolutional Neural Networks, training from scratch and fine-tuning existing models to develop baseline models
Experimenting with Transformer-based architectures, which may not necessarily outperform CNNs or even prove to be beneficial for this task, but can be a valuable experiment.
- **Tumor Segmentation**: Using Pixel-Wise classification models such as Unet, to classify MRI images to two regions of tumor and non-tumor.
Utilizing region based CNNs, such as Mask R-CNNs, for both classification and segmentation.
Comparing these two approaches with transformer-based segmentation models.

## Models


## Rationals
These models were chosen based on their proven effectiveness in image classification tasks, particularly in the medical field. .... are known for their performance in many computer vision applications, while they offer the flexibility to learn complex features from images.

## Results and Conclusion:
- Classification performance:
- Segmentation performance :

## Limitation:

## Setup:
The Models were trained and tested on a GPU enabled environmen t in Kaggle, ensuring efficient handling of the computationally intensive tasks.


## How to use:
- Download the dataset from the provided link.
- Download all notebook files in the same folder.
- Now, our files are ready to use.

## References:

https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms

