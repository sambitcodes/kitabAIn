# 📚 Book Recommendation System

A complete, production-ready **Book Recommendation System** built using
**Python, Pandas, Scikit‑learn, KNN**, and **Streamlit**.\
This system recommends books using two powerful models:

------------------------------------------------------------------------

## ⭐ Recommendation Models

### **1. Popularity-Based Recommendation (Weighted Rating)**

Perfect for **new users with no rating history**.
Recommends books that are:

-   Widely read
-   Highly rated
-   Rated by a large number of users

**Weighted rating formula used:**

```math
WR = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C
```

-  **v**: Number of ratings the book received
-  **m**: Minimum number of ratings required to be considered (e.g., 200)
-  **R**: Average rating of the book
-  **C**: Mean rating of all books

------------------------------------------------------------------------

### **2. Item-Based Collaborative Filtering (KNN Model)**

Recommends books similar to a given book by analyzing the behavior of
users who rated them.

-   Uses **cosine similarity**
-   Optimized with `scipy.sparse.csr_matrix`
-   Trained using **NearestNeighbors** from Scikit-learn
-   Based on dense subset:
    -   Users with ≥ 200 ratings
    -   Books with ≥ 50 ratings

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── artifacts/
    │   ├── cleaned_books.pkl
    │   ├── cleaned_ratings.pkl
    │   ├── cleaned_users.pkl
    │   ├── knn_model.pkl
    │   ├── pivot_df.pkl
    │   └── top_weighted_books.pkl
    │
    ├── data/
    │   ├── Books.csv
    │   ├── Ratings.csv
    │   └── Users.csv
    │
    ├── notebooks/
    │   ├── 1_Data_Preparation.ipynb
    │   ├── 2_EDA.ipynb
    │   ├── 3_Weighted_Rating.ipynb
    │   └── 4_Collaborative_Filtering.ipynb
    │
    ├── app.py
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🗂 Dataset

The project uses the **Book-Crossings Dataset** from Kaggle.

👉 Download here: *Kaggle link removed for packaging*
Unzip and place the following files in the `/data` folder:

-   `Books.csv`
-   `Ratings.csv`
-   `Users.csv`

------------------------------------------------------------------------

## 🧪 Methodology

### **1. Data Preparation (Notebook 1)**

-   Loads raw CSV files
-   Cleans column names
-   Fixes `year_of_publication`
-   Handles missing values in:
    -   `book_author`
    -   `publisher`
    -   `user_age`
-   Exports cleaned data as Pickle files

------------------------------------------------------------------------

### **2. Exploratory Data Analysis (Notebook 2)**

-   Uses Plotly for advanced visualizations
-   Analyzes:
    -   Rating distribution
    -   Most rated books
    -   Author frequency
    -   User demographics

------------------------------------------------------------------------

### **3. Weighted Rating Model (Notebook 3)**

-   Merges book and ratings data
-   Filters books with sufficient number of ratings
-   Calculates weighted rating
-   Saves `top_weighted_books.pkl`

------------------------------------------------------------------------

### **4. Collaborative Filtering (KNN Model) (Notebook 4)**

-   Filters dense data subset
-   Builds user‑item pivot table
-   Converts to sparse matrix
-   Trains KNN model using cosine similarity
-   Saves:
    -   `knn_model.pkl`
    -   `pivot_df.pkl`

------------------------------------------------------------------------

## ▶️ Running the Project

### **1. Clone the Repository**

``` bash
git clone <your-repo-url>
cd <repo-name>
```

------------------------------------------------------------------------

### **2. Create & Activate Virtual Environment**

``` bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

------------------------------------------------------------------------

### **3. Install Dependencies**

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### **4. Download & Place Dataset**

Place the Kaggle dataset CSVs into:

    /data/Books.csv
    /data/Ratings.csv
    /data/Users.csv

------------------------------------------------------------------------

### **5. Run the Notebooks**

Execute in order:

1.  `1_Data_Preparation.ipynb`\
2.  `2_EDA.ipynb`\
3.  `3_Weighted_Rating.ipynb`\
4.  `4_Collaborative_Filtering.ipynb`

This will generate model files inside `artifacts/`.

------------------------------------------------------------------------

### **6. Run the Streamlit App**

``` bash
streamlit run app.py
```

Your browser will automatically open the app.

------------------------------------------------------------------------

## 🎉 Features of Streamlit App

-   Multi‑tab interface
-   Popular book recommendations
-   Similar book recommendations
-   Searchable book list
-   Visual insights (optional if extended)

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome!

------------------------------------------------------------------------

## ❤️ Acknowledgements

-   Book‑Crossings Dataset
-   Kaggle
-   Scikit‑learn
-   Streamlit
