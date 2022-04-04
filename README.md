# Hippo-Net-Liver
DEEP LEARNING STAGING OF LIVER IRON CONTENT FROM MULTI-ECHO MR IMAGES
Background
MRI represents the most established approach for liver iron content (LIC) evaluation, but could be affected by software and readers induced variability.

Purpose
To develop a deep-learning method for unsupervised classification of LIC from magnitude multi-echo MR images.

Study Type
Retrospective.

Population/Subjects
In all, 1069 cases enrolled in the MIOT network from 2009 to 2021 were analyzed, including a training set (n = 885) and a test set (n = 184).

Field Strength/Sequence
1.5T, T2* multi echo magnitude images (in-phase/out-phase).

Assessment
An ensemble (HippoNet) of three deep-learning Convolutional Neural Network (CNN) networks (HippoNet-2D, HippoNet-3D, HippoNet-LSTM) was used to achieve unsupervised staging of LIC in thalassemia major patients using five classes. The training set was employed to construct the deep-learning model. The performance of the LIC staging model was evaluated in the test set. The model's performances were assessed by evaluating the accuracy, sensitivity, and specificity in respect to the ground truth labels and by comparison with operator-induced variability.

Statistical Tests
The Sklearn package in Python was used to evaluate the performances of the networks. The MedCalc software was used to compare the three network's performances by one-way and two-way analysis of variance-ANOVA. 

Results
The global accuracy of the ensemble deep-learning model was 0.90 against the 0.92 accuracy for the inter-observer variability. Accuracy, sensitivity, and specificity values for the five LIC classes were Normal: 0.96/0.93/0.97, Borderline: 0.95/0.85/0.98, Mild: 0.96/0.88/0.98, Moderate: 0.95/0.89/0.97, Severe: 0.97/0.95/0.98, respectively. Assessed LIC classes values are comparable with the inter-observer variability.

Data Conclusion
The proposed ensemble HippoNet network can perform unsupervised LIC staging and achieves good prognostic performance, comparable with the state-of-art clinical setting. 
