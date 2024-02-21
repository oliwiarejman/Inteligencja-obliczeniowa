import datetime
import math

def oblicz_biorytmy(dzien_zycia):
    fizyczny = round(math.sin(2 * math.pi * dzien_zycia / 23), 2)
    emocjonalny = round(math.sin(2 * math.pi * dzien_zycia / 28), 2)
    intelektualny = round(math.sin(2 * math.pi * dzien_zycia / 33), 2)
    return fizyczny, emocjonalny, intelektualny

def sprawdz_wyniki(fizyczny, emocjonalny, intelektualny):
    if fizyczny > 0.5 and emocjonalny > 0.5 and intelektualny > 0.5:
        return "Gratulacje! Dzisiaj masz dobry dzień!"
    elif fizyczny < -0.5 and emocjonalny < -0.5 and intelektualny < -0.5:
        return "To może być trudny dzień. Pamiętaj, że jutro będzie lepiej!"
    else:
        return "Dzisiaj masz standardowy dzień."

def sprawdz_nastepny_dzien(dzien_zycia):
    fizyczny, emocjonalny, intelektualny = oblicz_biorytmy(dzien_zycia + 1)
    if fizyczny > 0.5 or emocjonalny > 0.5 or intelektualny > 0.5:
        return "Nie martw się. Jutro będzie lepiej!"
    else:
        return "Następny dzień może być równie trudny."

imie = input("Podaj swoje imię: ")
rok = int(input("Podaj rok urodzenia (np. 1990): "))
miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
dzien = int(input("Podaj dzień urodzenia (1-31): "))

data_urodzenia = datetime.date(rok, miesiac, dzien)
dzis = datetime.date.today()
dzien_zycia = (dzis - data_urodzenia).days + 1

fizyczny, emocjonalny, intelektualny = oblicz_biorytmy(dzien_zycia)

print("\nWitaj", imie + "!")
print("Dziś jest", dzis.strftime("%Y-%m-%d"))
print("Numer dnia twojego życia:", dzien_zycia)
print("Twoje wyniki biorytmów:")
print("Fizyczny:", fizyczny)
print("Emocjonalny:", emocjonalny)
print("Intelektualny:", intelektualny)

print("\n" + sprawdz_wyniki(fizyczny, emocjonalny, intelektualny))
print(sprawdz_nastepny_dzien(dzien_zycia))
