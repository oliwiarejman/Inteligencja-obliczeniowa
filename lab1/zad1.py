import math
from datetime import datetime

def biorytmy():
    name = input("Podaj swoje imie: ")
    year = int(input("Podaj rok urodzenia: "))
    month = int(input("Podaj miesiac urodzenia(1-12): "))
    day = int(input("Podaj dzien urodzenia: " ))

    birthday = datetime(year, month, day)
    days = (datetime.now() - birthday).days

    yp = math.sin(((2*math.pi)/23)*days)
    ye = math.sin(((2*math.pi)/28)*days)
    yi = math.sin(((2*math.pi)/33)*days)
    yp2 = math.sin(((2*math.pi)/23)*(days+1))
    ye2 = math.sin(((2*math.pi)/28)*(days+1))
    yi2 = math.sin(((2*math.pi)/33)*(days+1))
    l1=[yp,ye,yi]
    l2=[yp2,ye2,yi2]

    print(f"Witaj {name}! Dzisiaj mija twój {days} dzień życia.")
    print(f"Twoje wyniki biorytmów prezentują sie następująco: ")
    for i in range (3):
        if i==0:
            print(f"Fizyczny: {yp}")
        elif i==1:
            print(f"Emocjonalny: {ye}")
        else:
            print(f"Intelektualny: {yi}")
        if l1[i]>0.5:
            print("Bardzo dobry wynik!")
        if l1[i]<-0.5:
            if l2[i]>l1[i]:
                print("Spoko, jutro bedzie lepiej")
            else:
                print("Spoko, jutro bedzie gorzej")


    return yp, ye ,yi,days


biorytmy()
#40min