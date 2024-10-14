import random

def generate_cartoon_script(summary):
    scenes = []
    summary_sentences = summary.split('.')

    for i, sentence in enumerate(summary_sentences):
        if sentence.strip():
            scene = {
                "scene_number": i + 1,
                "action": f"Character says: '{sentence.strip()}'",
                "background": f"Random scene {random.randint(1, 5)}",
                "emotion": random.choice(["happy", "sad", "angry", "neutral"]),
            }
            scenes.append(scene)

    cartoon_script = {
        "title": "Cartoon based on PDF",
        "scenes": scenes
    }

    return cartoon_script
