"""Create a small hard test set for full-utterance intent classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "our_part" / "data" / "hard_test"
OUTPUT_CSV = OUTPUT_DIR / "hard_test.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.json"


CHAT_ROWS = [
    ("chat_motion_explanation", "Can you explain how to wave politely without moving right now?"),
    ("chat_motion_explanation", "What does it mean when a robot raises its hand?"),
    ("chat_motion_explanation", "Tell me the difference between waving and pointing."),
    ("chat_motion_explanation", "Why do people bow in some cultures?"),
    ("chat_motion_explanation", "How should I describe a safe dance routine in words?"),
    ("chat_motion_explanation", "Could you define the phrase turn around in plain English?"),
    ("chat_motion_explanation", "Explain how a robot might balance before taking a step."),
    ("chat_motion_explanation", "What is the safest way for a machine to lift an object?"),
    ("chat_motion_explanation", "Describe the mechanics of a handshake without demonstrating it."),
    ("chat_motion_explanation", "Can you list common gestures used during a presentation?"),
    ("chat_motion_explanation", "What muscles do people use when they nod?"),
    ("chat_motion_explanation", "Explain why sudden arm movements can be unsafe near people."),
    ("chat_negation", "Do not move; just tell me what a bow means."),
    ("chat_negation", "Please stay still and explain how waving works."),
    ("chat_negation", "Do not raise your arm, only answer my question about stretching."),
    ("chat_negation", "Please do not dance; describe the history of dancing."),
    ("chat_negation", "Without moving, tell me how a robot could point to a door."),
    ("chat_negation", "Keep your hands down and explain what a thumbs-up means."),
    ("chat_negation", "Do not step forward; define forward motion for me."),
    ("chat_negation", "Please avoid gestures and summarize how to greet someone politely."),
    ("chat_negation", "Stay where you are and explain what it means to lean left."),
    ("chat_negation", "No demonstration is needed; just describe a salute."),
    ("chat_negation", "Do not act it out, but tell me how someone would kneel."),
    ("chat_negation", "Please remain still and explain why robots should move slowly."),
    ("chat_capability", "Can you tell me whether you are able to dance?"),
    ("chat_capability", "What kinds of motions can this robot perform?"),
    ("chat_capability", "Could you describe your arm movement limits?"),
    ("chat_capability", "What would happen if you tried to jump?"),
    ("chat_capability", "Can you explain whether you can point accurately?"),
    ("chat_capability", "Which gestures are safe for you to perform indoors?"),
    ("chat_capability", "Tell me what actions you know how to do."),
    ("chat_capability", "Can you describe how fast your hand can move?"),
    ("chat_capability", "What are your limitations when walking?"),
    ("chat_capability", "Could you explain the difference between a bow and a nod?"),
    ("chat_capability", "Can you talk about your balance system?"),
    ("chat_capability", "What motions should a humanoid avoid around children?"),
    ("chat_hypothetical", "If a robot waved at a crowd, what message would that send?"),
    ("chat_hypothetical", "If someone asks a robot to dance, what safety checks matter?"),
    ("chat_hypothetical", "When would pointing be considered impolite?"),
    ("chat_hypothetical", "Why might a robot refuse to lift a heavy box?"),
    ("chat_hypothetical", "If a person bows twice, how should I interpret it?"),
    ("chat_hypothetical", "What could go wrong if a robot spins too quickly?"),
    ("chat_hypothetical", "Suppose a robot claps during a meeting; is that appropriate?"),
    ("chat_hypothetical", "Why is it hard for robots to kneel smoothly?"),
    ("chat_hypothetical", "When should a robot avoid turning its body?"),
    ("chat_hypothetical", "What does a raised palm usually communicate?"),
    ("chat_hypothetical", "If a robot points at someone, could that seem rude?"),
    ("chat_hypothetical", "How would you explain a dance step to a beginner?"),
    ("chat_word_request", "Show me in words how to wave at a visitor."),
    ("chat_word_request", "Walk me through the idea of walking, but only verbally."),
    ("chat_word_request", "Give me a written explanation of how to salute."),
    ("chat_word_request", "Describe a jumping jack without doing one."),
    ("chat_word_request", "Can you answer with text how to make a thumbs-up?"),
    ("chat_word_request", "Tell me the steps of a bow in a sentence."),
    ("chat_word_request", "Use words to explain how someone turns left."),
    ("chat_word_request", "In words only, how would a robot greet a guest?"),
    ("chat_word_request", "Please describe the motion of clapping without clapping."),
    ("chat_word_request", "Can you explain how to sit down safely?"),
    ("chat_word_request", "Write a short description of a friendly wave."),
    ("chat_word_request", "Give me a verbal summary of how robots move their arms."),
]


MOTION_ROWS = [
    ("motion_direct_action", "Please wave once, but do it slowly."),
    ("motion_direct_action", "Raise your right hand for a moment."),
    ("motion_direct_action", "Turn your head toward the door."),
    ("motion_direct_action", "Point to the left side of the room."),
    ("motion_direct_action", "Give me a quick thumbs-up."),
    ("motion_direct_action", "Bow slightly to the visitor."),
    ("motion_direct_action", "Clap your hands two times."),
    ("motion_direct_action", "Take one small step backward."),
    ("motion_direct_action", "Nod once if you understand."),
    ("motion_direct_action", "Lean a little to your right."),
    ("motion_direct_action", "Stretch both arms outward."),
    ("motion_direct_action", "Face the window and pause."),
    ("motion_indirect_action", "Could you greet the visitor without saying anything?"),
    ("motion_indirect_action", "Make a small hello gesture for the camera."),
    ("motion_indirect_action", "Act out being surprised."),
    ("motion_indirect_action", "Use your arm to indicate yes."),
    ("motion_indirect_action", "Give the audience a silent goodbye."),
    ("motion_indirect_action", "Show approval without speaking."),
    ("motion_indirect_action", "Demonstrate a polite greeting."),
    ("motion_indirect_action", "Perform a tiny celebration."),
    ("motion_indirect_action", "Signal that the answer is correct."),
    ("motion_indirect_action", "Show me a friendly acknowledgement."),
    ("motion_indirect_action", "Make yourself look taller for a second."),
    ("motion_indirect_action", "Pretend you are looking for something on the floor."),
    ("motion_show_demonstrate", "Can you show me what a wave looks like?"),
    ("motion_show_demonstrate", "Show me how you would point at that chair."),
    ("motion_show_demonstrate", "Can you demonstrate a bow now?"),
    ("motion_show_demonstrate", "Show the movement for saying come here."),
    ("motion_show_demonstrate", "Can you act out a short dance step?"),
    ("motion_show_demonstrate", "Show me how you would stretch before exercise."),
    ("motion_show_demonstrate", "Demonstrate how you turn around."),
    ("motion_show_demonstrate", "Show how you would salute a guest."),
    ("motion_show_demonstrate", "Can you perform the gesture for stop?"),
    ("motion_show_demonstrate", "Show me the motion of opening your arms."),
    ("motion_show_demonstrate", "Could you demonstrate a simple nod?"),
    ("motion_show_demonstrate", "Act out lifting a light box."),
    ("motion_negation_trap", "Do not explain it; wave instead."),
    ("motion_negation_trap", "No words please, just point to the exit."),
    ("motion_negation_trap", "Instead of answering, bow to the class."),
    ("motion_negation_trap", "Skip the description and show the salute."),
    ("motion_negation_trap", "Do not tell me about dancing; dance once."),
    ("motion_negation_trap", "Stop talking and raise your hand."),
    ("motion_negation_trap", "Please do not describe the gesture, perform it."),
    ("motion_negation_trap", "No explanation is needed; clap once."),
    ("motion_negation_trap", "Without giving a lecture, turn left."),
    ("motion_negation_trap", "Rather than define a nod, nod now."),
    ("motion_negation_trap", "Do not summarize the action, just step forward."),
    ("motion_negation_trap", "Only move your arm; do not answer verbally."),
    ("motion_contextual", "The guest has arrived; welcome them physically."),
    ("motion_contextual", "The photo is starting; pose for it."),
    ("motion_contextual", "The music is playing; join in briefly."),
    ("motion_contextual", "The sign says stop; make that signal."),
    ("motion_contextual", "The audience applauded; respond with your hands."),
    ("motion_contextual", "The teacher asked for volunteers; indicate yourself."),
    ("motion_contextual", "The camera is on; give a friendly sign."),
    ("motion_contextual", "The visitor is leaving; say goodbye with your body."),
    ("motion_contextual", "The child copied you; make a simple safe gesture."),
    ("motion_contextual", "The box is in front of you; reach toward it."),
    ("motion_contextual", "The marker is on the table; indicate where it is."),
    ("motion_contextual", "The route turns right here; point that way."),
]


def write_csv(rows: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "utterance", "label", "hard_category"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, str]] = []
    for index, (category, utterance) in enumerate(CHAT_ROWS, start=1):
        rows.append(
            {
                "id": f"hard_chat_{index:03d}",
                "utterance": utterance,
                "label": "chat",
                "hard_category": category,
            }
        )
    for index, (category, utterance) in enumerate(MOTION_ROWS, start=1):
        rows.append(
            {
                "id": f"hard_motion_{index:03d}",
                "utterance": utterance,
                "label": "motion_query",
                "hard_category": category,
            }
        )

    write_csv(rows)
    manifest = {
        "name": "hard_test",
        "task": "full-utterance binary classification",
        "rows": len(rows),
        "labels": {
            "chat": len(CHAT_ROWS),
            "motion_query": len(MOTION_ROWS),
        },
        "design": [
            "chat rows intentionally include motion verbs, negation, capability questions, hypotheticals, and word-only requests",
            "motion_query rows include indirect actions, demonstrations, negation traps, and context-driven physical requests",
        ],
        "source": "manually curated in scripts/build_hard_test_dataset.py",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
