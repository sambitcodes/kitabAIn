import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="kitabAIn",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'knn_model.pkl')
PIVOT_PATH = os.path.join(ARTIFACTS_DIR, 'pivot_df.pkl')
BOOKS_PATH = os.path.join(ARTIFACTS_DIR, 'cleaned_books.pkl')
POPULAR_PATH = os.path.join(ARTIFACTS_DIR, 'top_weighted_books.pkl')
USERS_PATH = os.path.join(ARTIFACTS_DIR, 'cleaned_users.pkl')
RATINGS_PATH = os.path.join(ARTIFACTS_DIR, 'cleaned_ratings.pkl')

# --- Caching Functions ---

@st.cache_resource
def load_model():
    """Loads the trained KNN model."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please run the `4_Collaborative_Filtering.ipynb` notebook.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Loads all necessary dataframes from pickle files."""
    try:
        pivot_df = pd.read_pickle(PIVOT_PATH)
        books_df = pd.read_pickle(BOOKS_PATH)
        popular_df = pd.read_pickle(POPULAR_PATH)
        users_df = pd.read_pickle(USERS_PATH)
        ratings_df = pd.read_pickle(RATINGS_PATH)
        return pivot_df, books_df, popular_df, users_df, ratings_df
    except FileNotFoundError as e:
        st.error(f"Error: Data file not found. {e}. Please run all notebooks in order (1-4).")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# --- Helper Functions ---

def get_recommendations(book_name, model, pivot_df, books_df):
    """Gets top 5 book recommendations based on the KNN model."""
    recommendations = []
    try:
        # Get the index of the book
        book_index = np.where(pivot_df.index == book_name)[0][0]
        
        # Get distances and indices from the model
        distances, indices = model.kneighbors(pivot_df.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
        
        for i in range(1, 6):  # Start from 1 to skip the book itself
            idx = indices.flatten()[i]
            title = pivot_df.index[idx]
            distance = distances.flatten()[i]
            confidence = (1 - distance) * 100  # Convert distance to confidence percentage
            
            # Get book details from the main books dataframe
            details = books_df[books_df['book_title'] == title].drop_duplicates('book_title')
            
            if not details.empty:
                recommendations.append({
                    'title': title,
                    'author': details['book_author'].values[0],
                    'image': details['image_url_m'].values[0],
                    'confidence': f"{confidence:.2f}%"
                })
        return recommendations
    except IndexError:
        st.warning(f"The book '{book_name}' is in our list but was filtered out of the recommendation model. Try another book or check the 'Top 50 Popular' list.")
        return []
    except Exception as e:
        st.error(f"An error occurred during recommendation: {e}")
        return []

# --- EDA Plotting Functions (to be called in the app) ---
# These functions are copied from the EDA notebook for use in the app

def plot_rating_distribution(ratings_df):
    rating_counts = ratings_df['book_rating'].value_counts().sort_index()
    fig = px.bar(rating_counts, 
                 x=rating_counts.index, 
                 y=rating_counts.values, 
                 title='Distribution of All Book Ratings (0=Implicit, 1-10=Explicit)',
                 labels={'x': 'Rating', 'y': 'Count'},
                 color=rating_counts.index,
                 template='plotly_dark')
    return fig


def plot_book_ratings_count(ratings_df):
    fig = px.histogram(ratings_df, 
                       x='book_rating', 
                       nbins=10, 
                       title='Distribution of Book Ratings Count',
                       template='plotly_dark')
    return fig

def plot_user_ratings_count(ratings_df):
    fig = px.histogram(ratings_df, 
                       x='user_id', 
                       nbins=100, 
                       title='Distribution of User Ratings Count',
                       template='plotly_dark')
    return fig


def plot_top_authors(books_df):
    top_authors = books_df['book_author'].value_counts().head(20)
    fig = px.bar(top_authors, 
                 x=top_authors.values, 
                 y=top_authors.index, 
                 orientation='h', 
                 title='Top 20 Authors by Number of Books',
                 labels={'x': 'Number of Books', 'y': 'Author'},
                 template='plotly_dark')
    fig.update_yaxes(autorange="reversed")
    return fig

            
def plot_top_publishers(books_df):
    top_publishers = books_df['publisher'].value_counts().head(10)
    return px.bar(top_publishers, x=top_publishers.values, y=top_publishers.index, orientation='h', title='Top 10 Publishers by Book Count')
            
            
def plot_top_books_by_year(books_df):
    top_books = books_df.groupby('year_of_publication')['book_title'].count().sort_values(ascending=False).head(10)
    return px.bar(top_books, x=top_books.values, y=top_books.index, orientation='h', title='Top 10 Books by Year of Publication')
            


def plot_age_distribution(users_df):
    fig = px.histogram(users_df, 
                       x='age', 
                       nbins=50, 
                       title='Distribution of User Ages',
                       template='plotly_dark')
    return fig

def get_book_image_url(book_title, books_df):
    """Fetches the image URL for a given book title."""
    try:
        details = books_df[books_df['book_title'] == book_title].drop_duplicates('book_title')
        if not details.empty:
            return details['image_url_m'].values[0]
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching image URL: {e}")
        return None



# --- Main Application ---

# Load model and data
model = load_model()
pivot_df, books_df, popular_df, users_df, ratings_df = load_data()

# Check if data loading was successful
if model is None or pivot_df is None:
    st.error("Application cannot start. Please check the error messages above.")
else:
    left_head, right_head = st.columns([1,2])
    # with left_head:
        
    with right_head:
        # The "AI" part of the title should be in blue
        st.markdown(
    """
    <h1 style='text-align: left; color: white;'>
        📚 kitab<span style='color: #4169E1;'>AI</span>n 📚 
    </h1>
    """, 
    unsafe_allow_html=True
)

        st.markdown("Welcome! Get personalized book recommendations.")

    # --- Initialize Session State ---
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = ""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []

    def on_book_change():
        """Updates the session state with the newly selected book title immediately."""
        # Streamlit automatically updates the widget's key in session_state when a change occurs.
        # We retrieve that value and store it in our application's selected_book state variable.
        st.session_state.selected_book = st.session_state.book_selector
        # Optional: Clear recommendations if you want them to disappear until 'Recommend' is clicked again
        # st.session_state.recommendations = []

    # --- Create Tabs ---
    tab1, tab2 = st.tabs(["**Get Recommendations**", "**Explore The Data**"])

    # --- Tab 1: Recommendation ---
    with tab1:

        choice_left, choice_mid_a, choice_mid, choice_right = st.columns([2, 1, 1, 1.5])
        with choice_left:
            st.header("Find Your Next Favorite Book!")
            
            book_list = [""] + pivot_df.index.tolist()
            
            # Selectbox for book
            with st.container(border=True):
                selected_book = st.selectbox(
                        "Select a book you've enjoyed, and we'll find 5 similar books for you:",
                        book_list,
                        index=book_list.index(st.session_state.selected_book) if st.session_state.selected_book in book_list else 0,
                        key='book_selector', # This key is crucial for the callback function
                        on_change=on_book_change # ADDED: Call this function immediately on change
                    )

            # Buttons (Keep the existing button logic as is)
            col1, col2= st.columns([1, 1])
            with col1:
                # Note: The "Recommend" button is now just for getting recommendations, 
                # not for updating the book display itself.
                if st.button("🚀 Recommend", type="primary", use_container_width=True):
                    if st.session_state.selected_book: # Use the session state variable
                        with st.spinner(f"Finding books similar to '{st.session_state.selected_book}'..."):
                            st.session_state.recommendations = get_recommendations(st.session_state.selected_book, model, pivot_df, books_df)
                    else:
                        st.warning("Please select a book first.")
            
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.selected_book = ""
                    st.session_state.recommendations = []
                    st.rerun()

        # The display logic below this point already relies correctly on st.session_state.selected_book, 
        # so these columns will automatically update when st.session_state.selected_book changes.

        with choice_mid:
            # display selected book image
            with st.container(border=True):
                if st.session_state.selected_book:
                    image_url = get_book_image_url(st.session_state.selected_book, books_df)
                    if image_url:
                        st.image(image_url, width=150)
                    else:
                        st.warning("Image not available.")

        with choice_right:
            with st.container(border=True):
                if st.session_state.selected_book:
                    book = st.session_state.selected_book
                    st.success(f"{book}")
                    # Ensure books_df filtering is robust
                    book_info_row = books_df[books_df['book_title'] == st.session_state.selected_book]
                    if not book_info_row.empty:
                        book_info = book_info_row.iloc[0]
                    st.markdown(f"**Author:** {book_info['book_author']}")
                    st.markdown(f"**Publisher:** {book_info['publisher']}")
                    st.markdown(f"**Year of Publication:** {book_info['year_of_publication']}")
                    st.markdown(f"**ISBN:** {book_info['isbn']}")
                    
                else:
                    st.warning("Choose a book to see its details.")

        # st.divider()
        # Display Recommendations
        with st.container(border=True):
            if st.session_state.recommendations:
                st.subheader(f"Here are 5 books you might like:")
                cols = st.columns(5)
                for i, rec in enumerate(st.session_state.recommendations):
                    with cols[i]:
                        # Display confidence score in green color
                        st.image(rec['image'], width=150)
                        st.markdown(f"Confidence:<span style='color:green'> {rec['confidence']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**{rec['title']}**")
                        st.markdown(f"*by {rec['author']}*")
        
        # st.divider()
        
        # --- Top 50 Popular Books Section ---
        with st.expander("Don't know what to pick? Try one of these highly-rated fan favorites!"):
            st.header("Top 10 Popular Books")
            # st.markdown("Don't know what to pick? Try one of these highly-rated fan favorites!")
            
            if popular_df is not None:
                # Display as a formatted list/grid
                cols = st.columns(5)
                for i, row in popular_df.head(10).iterrows(): # Show top 10 as grid
                    with cols[i % 5]:
                        with st.container(border=True, height = 330):
                            st.image(row['image_url_m'], width=100, caption=f"Rating: {row['weighted_rating']:.2f}")
                            st.markdown(f"**{row['book_title']}**")
                            st.markdown(f"*by {row['book_author']}*")
            else:
                st.warning("Popular books data is not available.")

        with st.expander("List of Top 50 Popular Books (Highly rated)"):
            st.dataframe(popular_df[['book_title', 'book_author', 'weighted_rating', 'ratings_count', 'average_rating']].head(50))

    # --- Tab 2: Exploration ---
    with tab2:
        # st.header("Explore Our Book-Crossing Dataset")
        
        # --- Collapsed Section for EDA (remains full width) ---
        with st.expander("Show Insights from our Exploratory Data Analysis (EDA)"):
            st.subheader("Insights 💡📈📊🔍")
            
            if ratings_df is not None:
                st.plotly_chart(plot_rating_distribution(ratings_df), use_container_width=True)
                st.markdown("**Inference:** A vast majority of ratings are '0' (implicit ratings). Among explicit ratings (1-10), the distribution is skewed high, with '8' being the most common score.")

                #Add more insights
                st.plotly_chart(plot_book_ratings_count(ratings_df), use_container_width=True)
                st.markdown("**Inference:** Most books have a low rating count, with a long tail of books with higher rating counts. This suggests that while some books are highly rated, many others have received only a few ratings.")

                #Add more insights
                st.plotly_chart(plot_user_ratings_count(ratings_df), use_container_width=True)
                st.markdown("**Inference:** Most users have rated a low number of books, with a long tail of users who have rated a significant number of books. This could indicate that while some users are active book readers, many others have only started reading recently.")


            
            if books_df is not None:
                st.plotly_chart(plot_top_authors(books_df), use_container_width=True)
                st.markdown("**Inference:** Agatha Christie, William Shakespeare, and Stephen King are the most prolific authors, indicating a strong presence of classic and popular fiction.")

                #top 10 publishers by book count
                st.plotly_chart(plot_top_publishers(books_df), use_container_width=True)
                st.markdown("**Inference:** The top 10 publishers by book count are all major publishers in the United States, with 'Penguin Books' leading the pack.")
                
                #top 10 books by year of publication
                st.plotly_chart(plot_top_books_by_year(books_df), use_container_width=True)
                st.markdown("**Inference:** The top 10 books by year of publication are all classic novels, with 'The Lovely Bones' by Alice Sebold leading the pack.")

            if users_df is not None:
                st.plotly_chart(plot_age_distribution(users_df), use_container_width=True)
                st.markdown("**Inference:** The primary user base is in their 20s and 30s. The large spike around 30 is also influenced by our imputation of missing age data with the median.")


        with st.expander("Show Interactive Data Exploration Tools"):
            
            # --- Create the two-column layout for interactive plotting ---
            col_controls, col_plot_area = st.columns([1, 4]) # Adjust column ratios as needed (e.g., 1:3 for a large plot area)

            with col_controls:
                # Move all control widgets into the left column
                st.markdown("### Plot Settings ⚙️")

                # Select Dataset
                dataset_options = {"Books": books_df, "Users": users_df, "Ratings": ratings_df}
                dataset_name = st.selectbox("1. Choose a dataset to explore:", dataset_options.keys())
                
                df = dataset_options[dataset_name]
                
                # Initialize variables for plot generation outside the button press
                plot_type, x_col, y_col = None, None, None
                
                if df is not None:
                    # Select Plot Type
                    plot_type = st.selectbox("2. Choose a plot type:", ["Histogram", "Bar Chart (Top 50)", "Scatter Plot"])
                    
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    object_cols = df.select_dtypes(include='object').columns.tolist()

                    # Dynamic UI for column selection
                    try:
                        if plot_type == "Histogram":
                            x_col = st.selectbox("3. Select a numeric column:", numeric_cols)
                        elif plot_type == "Bar Chart (Top 50)":
                            x_col = st.selectbox("3. Select a categorical column:", object_cols + ['year_of_publication', 'age']) # Add some numeric as categorical
                        elif plot_type == "Scatter Plot":
                            x_col = st.selectbox("3. Select X-axis (numeric):", numeric_cols)
                            y_col = st.selectbox("4. Select Y-axis (numeric):", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                    except Exception as e:
                        st.warning(f"Could not populate column selectors: {e}")

                    # Explore Button - Keep the button here to trigger the generation in the right column
                    generate_plot_button = st.button("📊 Generate Plot", type="primary")

                else:
                    st.warning("Selected dataset is not loaded.")
                    generate_plot_button = False # Disable button if no data

            with col_plot_area:
                # This is where the plot will appear once the button is clicked
                if generate_plot_button and df is not None:
                    fig = None
                    inference = ""
                    with st.spinner("Generating plot..."):
                        try:
                            if plot_type == "Histogram" and x_col:
                                fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                                inference = f"This histogram shows the frequency distribution of **{x_col}**. The mean is **{df[x_col].mean():.2f}** and the median is **{df[x_col].median():.2f}**."
                            
                            elif plot_type == "Bar Chart (Top 50)" and x_col:
                                top_50 = df[x_col].value_counts().head(50)
                                fig = px.bar(top_50, x=top_50.index, y=top_50.values, title=f"Top 50 Values for {x_col}", labels={'x': x_col, 'y': 'Count'})
                                fig.update_xaxes(type='category')
                                inference = f"This bar chart displays the top 50 most frequent values in **{x_col}**. The most frequent value is **{top_50.index[0]}** with {top_50.values[0]} occurrences."

                            elif plot_type == "Scatter Plot" and x_col and y_col:
                                # Ensure statsmodels is installed for trendlines in plotly
                                try:
                                    fig = px.scatter(df.sample(n=min(5000, len(df))), x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}", trendline="ols")
                                    correlation = df[x_col].corr(df[y_col])
                                    inference = f"This scatter plot (showing a random sample of 5000 points) explores the relationship between **{x_col}** and **{y_col}**. The Pearson correlation coefficient is **{correlation:.3f}**."
                                except ValueError as ve:
                                    st.warning(f"Note: To use the 'trendline' feature, the `statsmodels` library is required. You can install it using `pip install statsmodels`.")
                                    fig = px.scatter(df.sample(n=min(5000, len(df))), x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col} (No Trendline)")
                                    correlation = df[x_col].corr(df[y_col])
                                    inference = f"This scatter plot explores the relationship between **{x_col}** and **{y_col}**. The Pearson correlation coefficient is **{correlation:.3f}**."


                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.subheader("Automated Inferences")
                                st.info(inference)
                            else:
                                st.warning("Please select valid columns to generate a plot.")
                                
                        except Exception as e:
                            st.error(f"Could not generate plot: {e}")
                elif generate_plot_button:
                    st.warning("Cannot generate plot. Selected dataset is invalid.")