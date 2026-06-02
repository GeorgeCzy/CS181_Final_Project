- # Proposal: Text / Motion Prompt Classification for Humanoid Robot Interaction

  ## Project Background

  In humanoid robot interaction, different user inputs require different response priorities.

  Some questions mainly require verbal responses:

  - “Who are you?”
  - “What is machine learning?”

  Other questions implicitly or explicitly expect robot motions:

  - “Can you dance?”
  - “Wave your hand.”
  - “Show me a happy pose.”

  The goal of this project is to build a lightweight intent classification model that determines whether a user query should trigger:

  1. A primarily text-based response (`text`)
  2. A motion-oriented response (`motion_prompt`)

  ---

  # Stage 1: Basic Intent Classification

  ## Core Idea

  Use a text embedding model to encode user input, then classify the embedding using a small MLP classifier.

  Pipeline:

  ```text
  User Input
      ↓
  Text Embedding Encoder
      ↓
  Sentence Embedding
      ↓
  MLP Classifier
      ↓
  {text / motion_prompt}
  ```

  ## Initial Model Design

  - Sentence embedding model:
    - Sentence-BERT
    - DistilBERT
    - Other lightweight encoder models

  - Classification head:
    - 2–3 layer MLP
    - Softmax output

  ## Dataset Design

  Two major categories:

  | Label         | Description                                 |
  | ------------- | ------------------------------------------- |
  | text          | Mainly conversational or informational      |
  | motion_prompt | Requires physical demonstration or gestures |

  Examples:

  | Input                      | Label         |
  | -------------------------- | ------------- |
  | “What is your name?”       | text          |
  | “Can you dance?”           | motion_prompt |
  | “Wave at me.”              | motion_prompt |
  | “Tell me what you can do.” | text          |

  Special attention should be paid to ambiguous boundary cases.

  ---

  # Stage 2: Early Intent Prediction for Real-Time Interaction

  ## Motivation

  In real conversations, speech arrives incrementally.

  Example:

  ```text
  “Can you…”
  “Can you dance…”
  “Can you dance for me?”
  ```

  Waiting for the full sentence introduces latency.

  The objective is to allow the robot to predict intent before the user finishes speaking.

  ---

  ## Key Idea: Temporal Masking for Early Intent Prediction

  To improve real-time interaction, the model should learn to classify intent using only the early portion of an utterance.

  We simulate the real interaction scenario where the robot only hears the first part of a sentence before making a decision.

  Example:

  Full input:

  ```text
  “Can you dance for me?”
  ```

  Training samples:

  | Input Portion Heard by Robot | Label         |
  | ---------------------------- | ------------- |
  | “Can you”                    | uncertain     |
  | “Can you dance”              | motion_prompt |
  | “Can you dance for”          | motion_prompt |

  Another example:

  ```text
  “Please wave your hand.”
  ```

  | Input Portion Heard by Robot | Label         |
  | ---------------------------- | ------------- |
  | “Please”                     | uncertain     |
  | “Please wave”                | motion_prompt |
  | “Please wave your”           | motion_prompt |

  The key idea is that the model only receives the temporally available part of the sentence, while the remaining future words are completely hidden.

  Goal:

  - Simulate real streaming human speech
  - Enable earlier intent prediction
  - Reduce robot response latency
  - Allow motion preparation before the user finishes speaking
  - Improve interaction naturalness

  ---

  # Evaluation Metrics

  Besides overall accuracy, evaluate:

  | Metric            | Description                            |
  | ----------------- | -------------------------------------- |
  | Full Accuracy     | Accuracy on complete sentences         |
  | Early Accuracy    | Accuracy using partial inputs          |
  | Decision Latency  | How early the model predicts correctly |
  | False Motion Rate | Incorrectly triggering motions         |
  | False Text Rate   | Missing motion-related requests        |

  ---

  # System Integration

  Final output should include:

  ```json
  {
    "label": "motion_prompt",
    "confidence": 0.91
  }
  ```

  Robot behavior can then be controlled using confidence thresholds.

  Example:

  | Condition              | Action                             |
  | ---------------------- | ---------------------------------- |
  | High motion confidence | Prepare motion generation pipeline |
  | High text confidence   | Use standard dialogue response     |
  | Low confidence         | Wait for more speech input         |

  ---

  # Expected Deliverables

  1. Annotated dataset
  2. Embedding + MLP baseline model
  3. Early prediction model
  4. Evaluation report
  5. Real-time demo system

  ---

  # Future Extensions

  Possible future directions:

  - Multi-class intent classification
  - Continuous motion intensity prediction
  - Joint text-motion generation
  - Integration with humanoid robot motion controllers
  - Streaming inference with ASR systems

  ---

  # Summary

  This project focuses on real-time intent understanding for humanoid robot interaction.

  The main contribution is not only intent classification itself, but also:

  - Early prediction under partial input
  - Low-latency inference
  - Real-time interaction optimization
  - Motion-trigger-aware dialogue understanding