# Predictia-categoriei-produsului-pe-baza-titlului
Task 3

## Descriere

Acest proiect folosește Machine Learning pentru a prezice categoria unui produs pe baza titlului acestuia.

Modelul analizează titlul produsului și folosește TF-IDF împreună cu mai multe caracteristici obținute din text pentru a clasifica produsele în una dintre cele 10 categorii.

## Setul de date

Setul de date conține informații despre produse, precum:

- Product Title
- Category Label
- Product ID
- Merchant ID
- Product Code
- Number of Views
- Merchant Rating
- Listing Date

Pentru antrenarea modelului au fost păstrate doar informațiile relevante pentru predicție.

Variabila țintă este:

`Category Label`

Categoriile din setul de date sunt:

- CPUs
- Digital Cameras
- Dishwashers
- Freezers
- Fridge Freezers
- Fridges
- Microwaves
- Mobile Phones
- TVs
- Washing Machines

## Curățarea datelor

În etapa de pregătire a datelor au fost realizate următoarele operații:

- standardizarea numelor coloanelor;
- eliminarea valorilor lipsă;
- standardizarea denumirilor categoriilor;
- eliminarea coloanelor care nu sunt relevante pentru predicție.

## Feature Engineering

Pe lângă titlul produsului, au fost create mai multe caracteristici:

- `title_char_count` – numărul de caractere din titlu;
- `title_word_count` – numărul de cuvinte din titlu;
- `has_number` – indică dacă titlul conține numere;
- `number_count` – numărul de cifre din titlu;
- `special_char_count` – numărul de caractere speciale;
- `has_special_char` – indică dacă titlul conține caractere speciale;
- `max_word_length` – lungimea celui mai lung cuvânt din titlu.

Aceste caracteristici au fost introduse pentru a oferi modelului informații suplimentare despre structura titlurilor produselor.

## TF-IDF

Titlul produsului a fost transformat în valori numerice folosind metoda TF-IDF.

Au fost folosite:

- lowercase pentru text;
- unigrams și bigrams;
- maximum 50.000 de caracteristici TF-IDF.

Caracteristicile numerice create prin feature engineering au fost scalate folosind `MinMaxScaler`.

## Modele testate

Au fost testate mai multe algoritmi de clasificare:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Linear Support Vector Machine (LinearSVC)

### Rezultate

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 0.96 | 0.96 |
| Naive Bayes | 0.91 | 0.91 |
| Decision Tree | 0.95 | 0.95 |
| Random Forest | 0.96 | 0.96 |
| **Linear SVM** | **0.97** | **0.97** |

Cel mai bun rezultat a fost obținut de **Linear SVM**, cu o acuratețe de aproximativ **97%**.

## Modelul final

Modelul final folosește următorul proces:

```text
Titlul produsului
        ↓
      TF-IDF
        +
Caracteristici numerice
        ↓
   MinMaxScaler
        ↓
    LinearSVC
        ↓
  Categoria produsului

Modelul antrenat este salvat in model/final_model.pkl

# Predicția unei categorii

Pentru a folosi modelul și a introduce un titlu de produs:
python predict_category.py

Programul va solicita un titlu:
Enter product title: iphone 7 32gb gold,4,3,Apple iPhone 7 32GB

și va afișa categoria prezisă:
Predicted category: Mobile Phones

Pentru închiderea programului se poate introduce:
exit


Modelul final, Linear SVM, a obținut aproximativ 97% accuracy pe setul de testare.

Concluzie

Experimentele au arătat că titlul produsului conține suficiente informații pentru a prezice cu o precizie ridicată categoria produsului.

Combinația dintre TF-IDF, feature engineering și Linear SVM a oferit cele mai bune rezultate dintre modelele testate.
