from pathlib import Path

SCRATCH_DIR = Path(__file__).resolve().parent.parent / "scratch"

DOCUMENTS = [
    {"path": SCRATCH_DIR / "hotel_policy.pdf", "source_type": "pdf"},
    {"path": SCRATCH_DIR / "room_rates.xlsx", "source_type": "xlsx"},
]

IN_SCOPE_QUESTIONS = [
    {"question": "What is the pet policy and fee?", "expected_filename": "hotel_policy.pdf"},
    {"question": "What time is check-in and check-out?", "expected_filename": "hotel_policy.pdf"},
    {
        "question": "What happens if I cancel my reservation within 48 hours of check-in?",
        "expected_filename": "hotel_policy.pdf",
    },
    {
        "question": "What deposit is required for group bookings of five or more rooms?",
        "expected_filename": "hotel_policy.pdf",
    },
    {"question": "How many points per dollar does the loyalty program earn?", "expected_filename": "hotel_policy.pdf"},
    {"question": "Are service animals exempt from the pet fee?", "expected_filename": "hotel_policy.pdf"},
    {"question": "What is the nightly rate for a Deluxe King room?", "expected_filename": "room_rates.xlsx"},
    {"question": "What is the maximum occupancy of the Family Suite?", "expected_filename": "room_rates.xlsx"},
    {"question": "Which room type has a roll-in shower for accessibility?", "expected_filename": "room_rates.xlsx"},
    {"question": "How much does the Executive Suite cost per night?", "expected_filename": "room_rates.xlsx"},
]

OUT_OF_SCOPE_QUESTIONS = [
    {"question": "What is the capital of France?"},
    {"question": "Who won the 2022 FIFA World Cup?"},
    {"question": "How do I reset my iPhone to factory settings?"},
    {"question": "What is the chemical formula for table salt?"},
    {"question": "Can you recommend a good recipe for banana bread?"},
    {"question": "What is the current stock price of Apple?"},
]
