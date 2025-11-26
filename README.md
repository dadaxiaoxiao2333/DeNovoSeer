DeNovoSeer: A Deep Learning Framework for Pathogenicity Prediction of De Novo Mutations

- This repository contains the implementation of DeNovoSeer, a deep learning framework for predicting the pathogenicity of de novo mutations using genomic annotation features.  The project includes modular pipelines for data preprocessing, model training, and result interpretation, enabling reproducible and extensible analysis across different datasets.


- The Data_preprocessing directory provides scripts for data cleaning and standardized feature preprocessing, ensuring consistent input format for downstream modeling.  The core model implementation is located in the SH-CNN directory, which includes a semi-supervised hybrid CNN-IDCNN architecture designed to capture both local and global patterns from mutation features.  The Visualization_and_Analysis directory includes tools for model interpretation and performance visualization.  SHAP-based feature attribution is supported to provide biological interpretability and enhance clinical transparency.


- This repository is structured for scalability and can be easily adapted to new datasets such as SPARK, Gene4Denovo, and other de novo mutation cohorts.  The code is compatible with GPU acceleration and supports both supervised and semi-supervised learning settings.  It is intended to serve as a framework for reproducible research in computational genomics and precision medicine.