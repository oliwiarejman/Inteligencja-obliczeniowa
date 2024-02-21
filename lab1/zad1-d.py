import math
from datetime import datetime

def biorhythms():
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia: "))
    month = int(input("Podaj miesiąc urodzenia (1-12): "))
    day = int(input("Podaj dzień urodzenia: "))

    birthday = datetime(year, month, day)
    days = (datetime.now() - birthday).days + 1

    yp = math.sin(((2 * math.pi) / 23) * days)
    ye = math.sin(((2 * math.pi) / 28) * days)
    yi = math.sin(((2 * math.pi) / 33) * days)
    yp_tomorrow = math.sin(((2 * math.pi) / 23) * (days + 1))
    ye_tomorrow = math.sin(((2 * math.pi) / 28) * (days + 1))
    yi_tomorrow = math.sin(((2 * math.pi) / 33) * (days + 1))

    print(f"Witaj {name}! Dzisiaj mija twój {days} dzień życia.")
    print("Twoje wyniki biorhythmów prezentują się następująco: ")

    categories = ["Fizyczny", "Emocjonalny", "Intelektualny"]
    results_today = [yp, ye, yi]
    results_tomorrow = [yp_tomorrow, ye_tomorrow, yi_tomorrow]

    for i in range(3):
        print(f"{categories[i]}: {results_today[i]}")
        if results_today[i] > 0.5:
            print("Bardzo dobry wynik!")
        elif results_today[i] < -0.5:
            if results_tomorrow[i] > results_today[i]:
                print("Spoko, jutro będzie lepiej.")
            else:
                print("Spoko, jutro będzie gorzej.")

    return yp, ye, yi, days

biorhythms()
