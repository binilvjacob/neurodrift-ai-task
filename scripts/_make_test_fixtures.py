"""One-off helper: generates a sample PDF and XLSX under scratch/ for verify_chunking.py smoke tests."""

from pathlib import Path

from fpdf import FPDF
from openpyxl import Workbook

OUT_DIR = Path(__file__).resolve().parent.parent / "scratch"
OUT_DIR.mkdir(exist_ok=True)

PARAGRAPHS = [
    "Section 1: Booking Policy. Guests may cancel a reservation free of charge up to 48 hours "
    "before the scheduled check-in time. Cancellations made within 48 hours of check-in are "
    "subject to a one-night penalty charged to the card on file. Group bookings of five rooms "
    "or more follow a separate policy described in Section 4.",
    "Section 2: Check-in and Check-out. Standard check-in begins at 3:00 PM local time and "
    "check-out is required by 11:00 AM. Early check-in and late check-out may be requested "
    "through the front desk and are granted subject to availability, sometimes at an additional "
    "fee depending on occupancy that day.",
    "Section 3: Pet Policy. Pets under 25 kg are welcome in designated pet-friendly rooms for a "
    "nightly fee of 20 dollars. Guests must notify the property at the time of booking. Service "
    "animals are exempt from this fee and from any weight restriction.",
    "Section 4: Group Bookings. Reservations of five or more rooms require a 25 percent deposit "
    "at the time of booking. The deposit is refundable if the group cancels at least 14 days "
    "before arrival. Group rates are not combinable with other promotional discounts.",
    "Section 5: Loyalty Program. Members of the rewards program earn 10 points per dollar spent "
    "on room charges. Points can be redeemed for free nights, starting at 10000 points for a "
    "standard room, subject to blackout dates during peak season.",
]


def make_pdf() -> Path:
    pdf = FPDF()
    for paragraph in PARAGRAPHS:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, paragraph)
    path = OUT_DIR / "hotel_policy.pdf"
    pdf.output(str(path))
    return path


def make_xlsx() -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "RoomRates"
    ws.append(["room_type", "nightly_rate_usd", "max_occupancy", "notes"])
    rows = [
        ("Standard Queen", 129, 2, "City view"),
        ("Standard Twin", 129, 2, "Two twin beds"),
        ("Deluxe King", 179, 2, "Ocean view"),
        ("Executive Suite", 259, 4, "Separate living area"),
        ("Family Suite", 299, 6, "Two bedrooms"),
        ("Accessible Queen", 129, 2, "Roll-in shower"),
    ]
    for row in rows:
        ws.append(row)
    path = OUT_DIR / "room_rates.xlsx"
    wb.save(str(path))
    return path


if __name__ == "__main__":
    pdf_path = make_pdf()
    xlsx_path = make_xlsx()
    print(f"wrote {pdf_path}")
    print(f"wrote {xlsx_path}")
