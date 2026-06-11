# Podsumowanie celu projektu

Ten projekt dotyczy przewidywania aktywnosci biologicznej czasteczek chemicznych z wykorzystaniem metod uczenia maszynowego i grafowych sieci neuronowych. Najbardziej prawdopodobnym celem jest zbudowanie modelu, ktory na podstawie struktury czasteczki oraz wybranych cech fizykochemicznych przewiduje jej aktywnosc wobec konkretnego targetu biologicznego.

## Glowny cel

Glownym celem projektu jest przewidywanie wartosci `pIC50` dla czasteczek chemicznych.

`IC50` oznacza stezenie zwiazku, przy ktorym obserwujemy 50% zahamowania aktywnosci danego targetu biologicznego, np. enzymu albo receptora. Im nizsze IC50, tym silniejszy zwiazek, bo mniejsze stezenie wystarcza do uzyskania efektu.

W modelowaniu zwykle uzywa sie `pIC50`, czyli logarytmicznej transformacji IC50:

```text
pIC50 = -log10(IC50 [M])
```

Dzieki temu problem jest wygodniejszy numerycznie. W tej skali wieksze `pIC50` oznacza zwykle silniejsza aktywnosc biologiczna.

## Co model ma przewidywac

Model ma dostawac opis czasteczki i zwracac jedna liczbe:

```text
wejscie: czasteczka
wyjscie: przewidywane pIC50
```

To jest problem regresji, a nie klasyfikacji. Nie przewidujemy etykiety typu aktywna/nieaktywna, tylko wartosc liczbowa opisujaca sile aktywnosci.

## Co jest wejscie do modelu

Projekt powinien wykorzystywac dwa typy informacji o czasteczce.

Pierwszy typ to cechy fizykochemiczne:

- masa czasteczkowa,
- polarna powierzchnia czasteczki,
- lipofilowosc,
- liczba akceptorow wiazan wodorowych,
- liczba donorow wiazan wodorowych,
- liczba rotowalnych wiazan.

Drugi typ to struktura czasteczki zapisana jako `SMILES`. To najwazniejszy element pod GNN. `SMILES` mozna przeksztalcic w graf:

- atomy sa wezlami,
- wiazania chemiczne sa krawedziami,
- cechy atomow i wiazan sa atrybutami grafu.

Dzieki temu model nie patrzy tylko na kilka ogolnych parametrow, ale moze uczyc sie z faktycznej struktury chemicznej czasteczki.

## Dlaczego target bialkowy jest wazny

Najwazniejsza rzecz biologiczna: aktywnosc IC50 zawsze dotyczy konkretnego targetu.

Ta sama czasteczka moze byc bardzo aktywna wobec jednego bialka i prawie nieaktywna wobec innego. Dlatego nie powinno sie mieszac pomiarow IC50 dla wielu roznych targetow w jednym prostym modelu regresyjnym.

Docelowo projekt powinien byc ustawiony tak:

```text
wybieramy jeden target bialkowy
zbieramy czasteczki z IC50 dla tego targetu
liczymy pIC50
trenujemy model przewidujacy pIC50 dla tego targetu
```

Wtedy model odpowiada na jasne pytanie:

```text
Jak aktywna bedzie dana czasteczka wobec wybranego bialka?
```

## Jakie modele chcemy porownac

Projekt powinien miec co najmniej prosty baseline i model grafowy.

Baseline:

- model klasyczny albo MLP,
- wejscie: cechy fizykochemiczne,
- cel: sprawdzic, jak dobrze da sie przewidywac `pIC50` bez pelnej struktury grafowej.

Model GNN:

- GCN albo GIN,
- wejscie: graf czasteczki zbudowany ze `SMILES`,
- cel: wykorzystac informacje o strukturze chemicznej.

Wariant rozszerzony:

- polaczyc embedding grafu z cechami fizykochemicznymi,
- na koncu uzyc warstw liniowych do regresji `pIC50`.

## Jak bedziemy oceniac wynik

Poniewaz to jest regresja, sensowne metryki to:

- MAE,
- RMSE albo MSE,
- R2.

Warto porownac wyniki na dwoch podzialach danych:

- random split,
- scaffold split.

Random split jest latwiejszy i moze dawac lepsze wyniki. Scaffold split jest trudniejszy, ale bardziej realistyczny, bo testuje generalizacje do nowych rodzin chemicznych.

## Do czego dazymy

Finalnie chcemy miec pipeline, ktory:

1. pobiera dane z ChEMBL,
2. wybiera jeden sensowny target bialkowy,
3. czysci i przygotowuje dane IC50,
4. przelicza IC50 na `pIC50`,
5. przygotowuje cechy fizykochemiczne i grafy czasteczek,
6. robi random split i scaffold split,
7. trenuje baseline,
8. trenuje model GNN,
9. porownuje wyniki,
10. pozwala opisac, czy struktura grafowa poprawia predykcje wzgledem prostych cech.

Najkrotsze streszczenie projektu:

```text
Budujemy model przewidujacy aktywnosc biologiczna czasteczek wobec jednego targetu bialkowego.
Targetem modelu jest pIC50.
Wejsciem sa cechy fizykochemiczne oraz struktura czasteczki jako graf.
Porownujemy prosty baseline z modelem GNN.
```

## Oczekiwany efekt koncowy

Na koniec projektu powinny powstac:

- przygotowany dataset dla jednego targetu,
- EDA pokazujace rozklad danych,
- baseline na cechach fizykochemicznych,
- model GNN na grafach czasteczek,
- porownanie wynikow na random split i scaffold split,
- krotkie wnioski o tym, czy wykorzystanie struktury czasteczki poprawia jakosc predykcji.

Projekt nie musi udowadniac, ze model jest gotowy do realnego odkrywania lekow. Powinien natomiast pokazac poprawny proces: sensowny dataset, uzasadnione czyszczenie danych, dobry podzial train/test, baseline, model grafowy i uczciwa ewaluacja.
