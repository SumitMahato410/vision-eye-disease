"""Plain-language information and prevention guidance for each class.

This is the "& Prevention" half of the project title. The model output alone is
just a word and a number; what makes the tool useful to someone in a village
with no nearby ophthalmologist is knowing what to do next.

Everything here is general health education, not personalised medical advice.
``urgency`` drives the colour of the banner in the UI:
    routine   — see an eye doctor at your convenience
    soon      — book an appointment within days to weeks
    urgent    — seek care today
"""

from __future__ import annotations

DISCLAIMER = (
    "This is an automated screening aid, not a diagnosis. It can be wrong. "
    "Only a qualified eye-care professional can diagnose an eye condition. "
    "If your vision is changing, or your eye is painful, seek care immediately "
    "regardless of what this tool says."
)

_UNKNOWN = {
    "label": "Unrecognised",
    "urgency": "routine",
    "summary": "This condition is not in the current reference list.",
    "prevention": [],
    "next_steps": ["Show the photo and this result to an eye-care professional."],
}

ADVICE: dict[str, dict] = {
    # ---------------- Internal / fundus ----------------
    "diabetic_retinopathy": {
        "label": "Diabetic retinopathy",
        "urgency": "soon",
        "summary": (
            "Damage to the small blood vessels of the retina caused by long-term high "
            "blood sugar. It is often painless and causes no symptoms until it is "
            "advanced, which is exactly why screening matters."
        ),
        "prevention": [
            "Keep blood glucose in the range your doctor sets for you; HbA1c control "
            "is the single biggest factor in slowing progression.",
            "Control blood pressure and cholesterol — both accelerate retinal damage.",
            "Have a dilated retinal examination at least once a year if you are diabetic.",
            "Stop smoking; it compounds small-vessel damage.",
        ],
        "next_steps": [
            "Book a dilated eye examination with an ophthalmologist.",
            "Ask your physician to review your diabetes control.",
            "Go the same day if you notice sudden floaters, dark patches or vision loss.",
        ],
    },
    "glaucoma": {
        "label": "Glaucoma (suspected optic disc changes)",
        "urgency": "soon",
        "summary": (
            "A group of conditions that damage the optic nerve, usually linked to raised "
            "pressure inside the eye. Vision lost to glaucoma cannot be recovered, but "
            "treatment started early can stop further loss."
        ),
        "prevention": [
            "Get eye pressure checked regularly after age 40, and earlier if a close "
            "relative has glaucoma.",
            "Use prescribed eye drops exactly as directed — skipped doses are the most "
            "common reason treatment fails.",
            "Protect the eyes from injury and keep blood pressure under control.",
        ],
        "next_steps": [
            "See an ophthalmologist for tonometry, a visual field test and an optic "
            "nerve examination — a photo alone cannot confirm glaucoma.",
            "Mention any family history of glaucoma at the appointment.",
        ],
    },
    "cataract": {
        "label": "Cataract",
        "urgency": "routine",
        "summary": (
            "Clouding of the natural lens of the eye, usually age-related. It develops "
            "slowly and is treated very effectively with routine surgery."
        ),
        "prevention": [
            "Wear UV-blocking sunglasses outdoors.",
            "Avoid smoking and limit alcohol.",
            "Manage diabetes, which brings cataracts on earlier.",
        ],
        "next_steps": [
            "See an ophthalmologist to assess how much the lens clouding affects daily "
            "activities such as reading and night driving.",
            "Surgery is usually planned, not emergency — there is time to think about it.",
        ],
    },
    "amd": {
        "label": "Age-related macular degeneration",
        "urgency": "soon",
        "summary": (
            "Deterioration of the macula, the central part of the retina used for sharp "
            "detail. It affects central vision while leaving peripheral vision intact."
        ),
        "prevention": [
            "Do not smoke — smoking is the strongest modifiable risk factor.",
            "Eat leafy greens and oily fish; keep blood pressure controlled.",
            "Wear sunglasses and have regular retinal checks after age 55.",
        ],
        "next_steps": [
            "See a retina specialist. The wet form is treatable with injections, and "
            "outcomes depend heavily on starting early.",
            "Seek care urgently if straight lines look wavy or a blurred patch appears "
            "in the centre of your vision.",
        ],
    },

    # ---------------- External / close-up ----------------
    "conjunctivitis": {
        "label": "Conjunctivitis (pink eye)",
        "urgency": "soon",
        "summary": (
            "Inflammation of the clear membrane covering the white of the eye, from a "
            "virus, bacteria or an allergy. Most cases settle on their own, but the "
            "infectious kinds spread very easily."
        ),
        "prevention": [
            "Wash hands often and avoid touching or rubbing the eyes.",
            "Do not share towels, pillowcases or eye makeup.",
            "Stop wearing contact lenses until the eye is completely better, and replace "
            "the lenses and case.",
        ],
        "next_steps": [
            "See a doctor if there is pain, light sensitivity, thick discharge or any "
            "change in vision — those suggest something more serious than simple "
            "conjunctivitis.",
            "Cool compresses and lubricating drops ease the irritation meanwhile.",
        ],
    },
    "stye": {
        "label": "Stye / eyelid inflammation",
        "urgency": "routine",
        "summary": (
            "A blocked, infected oil gland at the eyelid margin. It shows up as a tender "
            "red lump and usually resolves within a week or two."
        ),
        "prevention": [
            "Clean the eyelid margins gently and regularly, especially if you get styes "
            "repeatedly.",
            "Remove eye makeup before sleeping and replace old mascara.",
            "Wash hands before handling contact lenses.",
        ],
        "next_steps": [
            "Apply a warm compress for 10 minutes, several times a day. Do not squeeze it.",
            "See a doctor if it does not improve in two weeks, keeps returning, or the "
            "redness spreads across the eyelid or cheek.",
        ],
    },
    "redness": {
        "label": "Eye redness / irritation",
        "urgency": "routine",
        "summary": (
            "Visible redness of the white of the eye. It has many causes — dryness, "
            "screen strain, allergy, irritants, infection — and the photo alone cannot "
            "distinguish between them."
        ),
        "prevention": [
            "Follow the 20-20-20 rule at screens: every 20 minutes, look 20 feet away "
            "for 20 seconds.",
            "Use lubricating drops for dryness and keep away from smoke and dust.",
            "Do not overwear contact lenses or sleep in them.",
        ],
        "next_steps": [
            "If redness lasts more than a few days, or comes with pain or blurred vision, "
            "have it examined.",
        ],
    },
    "normal": {
        "label": "No obvious disease detected",
        "urgency": "routine",
        "summary": (
            "The model did not find features it associates with the conditions it was "
            "trained on. This is not a clean bill of health: early disease often looks "
            "normal in a photograph, and the model only knows a handful of conditions."
        ),
        "prevention": [
            "Have a full eye examination every one to two years, annually if you are "
            "diabetic or over 60.",
            "Wear UV protection outdoors and take regular breaks from screens.",
            "Control blood sugar and blood pressure.",
        ],
        "next_steps": [
            "Keep to routine check-ups.",
            "See a professional about any symptom you actually have, regardless of this "
            "result.",
        ],
    },
}


def get(class_name: str) -> dict:
    entry = ADVICE.get(class_name.lower().strip())
    if entry is None:
        return {**_UNKNOWN, "label": class_name.replace("_", " ").title()}
    return entry


def pretty(class_name: str) -> str:
    return get(class_name)["label"]
