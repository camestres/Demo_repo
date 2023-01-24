Project description
--------------------

Here is a solution for Kaggle competition [Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection/overview/description).
The objective of this task is to predict keypoint positions on face images. 
Submissions are scored on the root mean squared error (RMSE).

The solution approach includes next steps:
1. Train a baseline solution (Eff. B0, no augmentation). It leads to RMSE approximately 2.10-2.20
2. Train the baseline + augmantation. RMSE ~ 1.86
3. Split the data and train two models (4 & 15 landmarks). RMSE ~ 1.71

Data for this project is here [Kaggle dataset]([https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/competitions/facial-keypoints-detection/data)).
