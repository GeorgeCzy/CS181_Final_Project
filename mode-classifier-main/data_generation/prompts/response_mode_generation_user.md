Generate {batch_size} new examples for the response-mode classifier.

Required label counts in this batch:

- `chat`: {chat_count}
- `motion_query`: {motion_count}

Coverage goals for this batch:

- Everyday small talk and factual questions
- Explanations, recommendations, planning, emotional support, and troubleshooting
- Robot capability questions that should remain `chat`
- Explicit movement requests, gestures, demonstrations, pointing, gaze/orientation, locomotion, fetching, object manipulation, safety stop/stay commands, and nonverbal answers
- Ambiguous boundary cases where the label is still clear from wording

Reference examples:

{reference_examples}

Output schema:

{{
  "examples": [
    {{
      "utterance": "English human utterance",
      "label": "chat"
    }}
  ]
}}

Constraints:

- Use only the labels `chat` and `motion_query`.
- Each utterance must be one sentence or one short user turn.
- Do not include IDs; the script will assign IDs.
- Do not include scenario names, explanations, comments, or metadata.
- Do not repeat reference examples verbatim.
- Do not repeat examples within the batch.

