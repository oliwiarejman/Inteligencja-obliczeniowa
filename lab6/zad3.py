import cv2
import os

def policz_ptaki_na_obrazie(obraz):
    obraz_szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    
    krawedzie = cv2.Canny(obraz_szary, 50, 150)
    
    kontury, _ = cv2.findContours(krawedzie, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    liczba_ptakow = len(kontury)
    
    return liczba_ptakow

sciezka_folderu = "bird_miniatures"

lista_plikow = os.listdir(sciezka_folderu)

for plik in lista_plikow:
    if plik.endswith(".jpg") or plik.endswith(".png"):
        obraz = cv2.imread(os.path.join(sciezka_folderu, plik))
        
        if obraz is not None:
            liczba_ptakow = policz_ptaki_na_obrazie(obraz)
            
            print(f"Obraz {plik}: Liczba ptaków = {liczba_ptakow}")
        else:
            print(f"Nie można wczytać obrazu: {plik}")
