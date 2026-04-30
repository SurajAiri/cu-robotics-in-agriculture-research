# Architecture Overview

This project has two connected layers:

1. Phase I builds the trained regression pipelines from agricultural data.
2. Phase II uses those production pipelines to recommend environment adjustments for a crop field.

## High-Level Flow

```text
Raw crop yield dataset
    ↓
Data cleaning and validation
    ↓
Feature engineering and preprocessing
    ↓
Model training and hyperparameter tuning
    ↓
Model evaluation and selection
    ↓
Production artifacts in models/production
    ↓
Streamlit phase-II app
    ↓
Current field inputs → environment variations → yield prediction → best configuration
    ↓
Robot-friendly recommendations
```

## Mermaid Diagram

```mermaid
flowchart TD
    A[Raw agricultural data\nKaggle crop yield dataset] --> B[Cleaning and validation]
    B --> C[Processed datasets\ndata/processed]
    C --> D[Feature engineering and preprocessing]
    D --> E[Model training and tuning\nscikit-learn, XGBoost, CatBoost]
    E --> F[Model evaluation\nRMSE, MAE, R2]
    F --> G[Production artifacts\nmodels/production]
    G --> H[Phase-II Streamlit app\napp.py]
    H --> I[User enters current field conditions]
    I --> J[Generate environmental variations]
    J --> K[Predict yield for each configuration]
    K --> L[Select optimal configuration]
    L --> M[Display robot recommendations]

    subgraph Phase I[Phase I: Model Training]
        A
        B
        C
        D
        E
        F
        G
    end

    subgraph Phase II[Phase II: Robotic Decision Support]
        H
        I
        J
        K
        L
        M
    end
```

## System Boundaries

- Phase I is experimental and focuses on building the best yield prediction model.
- Phase II is a decision-support layer that consumes the trained model outputs.
- The robot actions shown in the phase-II concept are recommendations, not a verified autonomous control loop.

## Key Runtime Inputs

- Crop type
- Season
- State
- Area
- Annual rainfall
- Fertilizer per area
- Pesticide per area

The app loads the champion model by default and keeps a rollback model available for comparison or fallback.
