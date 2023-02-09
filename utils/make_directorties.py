import os

path = os.path.join(os.getcwd(), "../resources")

names = [
    "Direction_indicator_left",
    "Differential",
    "Forward Traction",
    "lights",
    "Signalisation_one",
    "Signalisation_two",
    "Forward_suspention",
    "Battery_indicator",
    "direction_indicator_right",
    "Preheating",
    "Oil_Pressure",
    "Automatic_PDF",
]

print(path)

for i, n in enumerate(names):
    os.mkdir(os.path.join(path, f"{i + 1}.{n}"))
    print(f"{n} created")
