"""
generate_data.py
-----------------
Generates a synthetic but realistic dataset of civic complaints used to
train the category-classification model. In a real deployment you would
replace this with actual historical complaint records from your city's
grievance portal (e.g., NYC 311, Swachhata App, or a local municipal API).
"""

import csv
import random

random.seed(42)

AREAS = [
    "MG Road", "Gandhi Nagar", "Patel Nagar", "Lake View Colony",
    "Ring Road", "Old Town", "Riverside", "Industrial Estate",
    "Green Park", "Central Market", "Hill View", "Sector 12",
]

# category -> list of template sentences with {area} placeholder
TEMPLATES = {
    "Roads": [
        "There is a huge pothole on the main road near {area} causing accidents.",
        "The road near {area} is completely damaged after the rains, very hard to drive.",
        "Speed breakers on {area} road are broken and unmarked, dangerous at night.",
        "Road near {area} has not been repaired for months, full of craters.",
        "Footpath near {area} is broken and unsafe for pedestrians.",
        "Newly laid road near {area} is already cracking and full of water.",
    ],
    "Water Supply": [
        "No water supply in {area} for the last three days.",
        "Water pipeline near {area} is leaking and wasting a lot of water.",
        "The water coming from the tap in {area} is muddy and smells bad.",
        "Low water pressure in {area}, tank not filling up properly.",
        "Contaminated water supply reported in {area}, people are falling sick.",
        "Irregular water supply timings in {area}, no prior notice given.",
    ],
    "Electricity": [
        "Frequent power cuts in {area} for the past week.",
        "Street transformer near {area} is sparking, very dangerous.",
        "Electric pole near {area} is leaning and about to fall.",
        "Exposed electrical wires near {area}, risk of electrocution.",
        "Voltage fluctuation in {area} is damaging home appliances.",
        "No power supply in {area} since morning, no update from department.",
    ],
    "Garbage": [
        "Garbage has not been collected in {area} for over a week.",
        "Overflowing garbage bin near {area} creating a health hazard.",
        "Illegal dumping of waste near {area} attracting stray animals.",
        "Bad smell from uncollected trash near {area} market.",
        "Garbage truck skipped {area} street again this week.",
        "Plastic waste burning near {area}, causing air pollution.",
    ],
    "Drainage": [
        "Drainage near {area} is blocked and overflowing onto the road.",
        "Sewage water stagnating near {area} due to broken drain.",
        "Open drain near {area} is a safety hazard for children.",
        "Drain overflow near {area} flooding houses during rain.",
        "Foul smell from clogged drainage system in {area}.",
        "Manhole cover missing near {area}, very dangerous at night.",
    ],
    "Street Lighting": [
        "Street lights near {area} have not worked for two weeks.",
        "{area} road is completely dark at night, no functioning streetlights.",
        "Streetlight pole near {area} is damaged and hanging loose.",
        "New streetlights needed near {area}, area is unsafe after dark.",
        "Flickering streetlights near {area} need urgent repair.",
    ],
    "Public Safety": [
        "Stray dogs near {area} are attacking pedestrians, urgent action needed.",
        "Unauthorized construction near {area} blocking the road, safety risk.",
        "Broken boundary wall near {area} school poses risk to children.",
        "Illegal parking near {area} is blocking emergency vehicle access.",
        "Fire safety equipment missing in {area} public building.",
        "Loose electrical cables hanging near {area} bus stop, safety hazard.",
    ],
    "Parks & Public Spaces": [
        "Playground equipment in {area} park is broken and rusted.",
        "{area} park is poorly maintained, overgrown with weeds.",
        "No proper lighting in {area} park, unsafe for evening walkers.",
        "Public toilets near {area} park are unhygienic and need cleaning.",
        "Benches in {area} park are damaged and need replacement.",
    ],
}

URGENT_WORDS = [
    "urgent", "dangerous", "immediately", "emergency", "accident",
    "risk", "hazard", "fire", "electrocution", "sick", "unsafe",
]

rows = []
for category, templates in TEMPLATES.items():
    for template in templates:
        # create several variations per template using different areas
        for area in random.sample(AREAS, k=6):
            text = template.format(area=area)
            rows.append({"text": text, "category": category})

random.shuffle(rows)

with open("data/sample_complaints.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "category"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} synthetic complaints across {len(TEMPLATES)} categories.")
