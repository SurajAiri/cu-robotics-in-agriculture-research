# Phase II: Robotic Decision Support and Environment Optimization

Phase II is the application layer. It takes the trained production model from Phase I and uses it to evaluate possible field-condition adjustments before recommending the best configuration to the user.

## Concept

The idea is to simulate a robot-assisted optimization loop:

1. Read the current field conditions.
2. Generate a set of possible environment variations.
3. Predict crop yield for each candidate configuration.
4. Select the configuration with the highest predicted yield.
5. Present the resulting actions as robot recommendations.

## Inputs Used in the App

The Streamlit app in [app.py](../app.py) collects:

- Crop type
- Season
- State
- Area
- Annual rainfall
- Fertilizer per area
- Pesticide per area

It also allows switching between the champion and rollback production pipelines.

## Optimization Variables

The current implementation explores variations in:

- Fertilizer per area
- Pesticide per area
- Annual rainfall, treated as an irrigation-equivalent control

For each parameter, the app builds a small grid of candidate values around the current input, then scores all combinations with the trained model.

## Decision Flow

```text
Current field conditions
    ↓
Generate environment variations
    ↓
Predict yield for every configuration
    ↓
Choose the maximum predicted yield
    ↓
Compute recommended action deltas
    ↓
Display robot-facing guidance
```

## Output

The app displays:

- Current predicted yield
- Optimal predicted yield
- Percentage improvement
- Recommended increases or decreases for irrigation, fertilizer, or pesticide
- A detailed table of candidate configurations

## Conceptual Robot Actions

The boxed section in the handwritten sketch is intentionally conceptual. It describes the expected future behavior of an agricultural robot, such as:

- Increasing irrigation
- Adjusting fertilizer application
- Adjusting pesticide usage

Those actions are not experimentally validated in this repository. The current implementation stops at recommendation generation.

## Scope Boundary

This phase uses the trained yield model as a decision-support engine. It does not directly actuate robot hardware, and it does not prove that the recommended actions are safe or optimal in a live deployment without additional field validation.
