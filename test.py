from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# Define label dimensions
width = 63.5 * mm  # Replace with your desired width
height = 38.1 * mm  # Replace with your desired height

# Create a PDF document and canvas
c = canvas.Canvas("label.pdf", pagesize=(width, height))

# Define text elements and styles
text_elements = [
    ("Onion CC", 14, "Helvetica-Bold", (10, height - 5)),
    ("Trialnumber: 0512-8", 10, "Helvetica", (10, height - 15)),
    ("Date: dinsdag 5/dec/23", 10, "Helvetica", (10, height - 20)),
    ("Move to cel:", 10, "Helvetica-Bold", (10, height - 28)),
    ("PGO1 Stelling:", 10, "Helvetica", (20, height - 33)),
    ("Planted by:", 10, "Helvetica", (10, height - 43)),
    ("First watering:", 10, "Helvetica", (10, height - 48)),
    ("Direct", 10, "Helvetica", (40, height - 48)),
    ("Lights on:", 10, "Helvetica", (10, height - 53)),
    ("Zodra planten boven grond komen", 10, "Helvetica", (30, height - 53)),
    ("Rooien:", 10, "Helvetica", (10, height - 63)),
    ("woensdag 7/feb/24", 10, "Helvetica", (25, height - 63)),
    ("Final Count:", 10, "Helvetica", (10, height - 70)),
    ("zondag 11/feb/24", 10, "Helvetica", (35, height - 70)),
]

# Draw text elements on the canvas
for text, font_size, font_name, position in text_elements:
    c.setFont(font_name, font_size)
    c.drawString(*position, text)

# Save the PDF document
c.save()

print("Label generated successfully as label.pdf")
