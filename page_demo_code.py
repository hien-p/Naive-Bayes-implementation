import streamlit as st
from PIL import Image

import pandas as pd
# Import function to fetch dataset
from sklearn.datasets import load_iris
# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import classification_report, confusion_matrix


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Import custom latex display and numbering class
#from latex_equation_numbering import latex_equation_numbering

# state = _get_state()

st.page_config = st.set_page_config(
    page_title="Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
    
}

footer{
    visibility:hidden;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

def button_created(key):
    """
    Callback function for initializing buttons in streamlit
    """
    if key+'_dict' not in st.session_state:
        st.session_state[key+'_dict'] = {'was_created': True, 'was_pressed': False}



col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.title(f'Our  app demo {st.__version__}')

st.header("Naive-Bayes with sklearn".upper())
st.info("This page includes two interactive examples of analysing and classifying datasets using the tools available in sklearn")
st.write("For our examples, we will use datasets already included with sklearn in the sklearn.datasets module. A list of these, with more details on each, can be found [here](https://scikit-learn.org/stable/datasets.html)")
st.markdown(""" The examples on this page include the **Iris dataset**, which contains fully numerical data.  The Iris dataset contains features that describe the physical appearance of **150 samples** of three species belonging to the iris genus of flowering plant. 
The Iris dataset contains features that describe the physical appearance of 150 samples of three species belonging to the iris genus of flowering plant. Since this dataset consists of raw text, we will need to do some preprocessing in order to feed it into a classifier to predict categories.
""")
image = Image.open('data/iris.jpeg')
st.image(image, caption='Three species of iris in the iris dataset')
st.header("Iris datasets")

st.markdown("### Loading the Data")


# To load the Iris dataset, we use the load_iris() function from the sklearn.datasets module
st.markdown(" To load the **Iris dataset**, we use the  `load_iris()` function from the  `sklearn.datasets` module")


st.markdown("""
```python
from sklearn.datasets import load_iris

iris = load_iris()
```
The abreviations `pd` and `sns` are conventions. By using the `as_frame=True` keyword argument in the `load_iris()` function call, `pandas` DataFrame and Series are returned for the feature data and target respectfully. 
""")
st.write("Run the code to view the feature and target data, and a statistics summary of the data features.")

run_button_key = 'iris_load_run_button'
code_col, output_col = st.columns(2)
with code_col:
    st.subheader('Code:')
    st.write(
        '''
        ```python
        from sklearn.datasets import load_iris

        iris = load_iris()

        print('Feature names:')
        print(iris.feature_names)
        
        print('Target names:')
        print(iris.target_names)

        print('Data description:')
        print(iris.DESCR)

        print('First three samples:')
        print('Feature values:')
        print(iris.data[:3])
        print('Target values:')
        print(iris.target[:3])
        ```
        '''
    )
with output_col:
    st.subheader('Output:')
run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
if run_button :
    #st.session_state[run_button_key+'_dict']['was_pressed'] = True

    iris = load_iris()

    with output_col:
        st.write('**Feature names:**')
        st.text(iris.feature_names)
        
        st.write('**Target names:**')
        st.text(iris.target_names.tolist())
        
        st.write('**Data description:**')
        with st.expander('Expand description'):
            st.text(iris.DESCR)
        st.text('\n  ')
        
        st.write('**First three samples:**')
        st.text('Feature values:')
        st.text(iris.data[:3].tolist())
        st.text('Target values:')
        st.text(iris.target[:3].tolist())



# -- Exploring the Data'

st.write(' ### Exploring the Data')
st.write(
    '''
    Let's do some exploratory data analysis (EDA) for this dataset before jumping into training classifiers. To facilitate this process, we will use `pandas` to create a DataFrame of the iris dataset, and use `seaborn` for visualizations. Both packages can be imported using
    ```python
    import pandas as pd
    import seaborn as sns
    ```
    The abreviations `pd` and `sns` are conventions. By using the `as_frame=True` keyword argument in the `load_iris()` function call, `pandas` DataFrame and Series are returned for the feature data and target respectfully. Run the code to view the feature and target data, and a statistics summary of the data features.
    '''
)
run_button_key = 'iris_stats_run_button'
code_col, output_col = st.columns(2)
with code_col:
    st.write('#### Code:')
    st.write(
        '''
        ```python   
        import pandas as pd
        from sklearn.datasets import load_iris

        iris_df = load_iris(as_frame=True)

        print('iris_df.data dataframe:')
        print(iris_df.data)

        print('iris_df.target dataframe:')
        print(iris_df.target)

        print('Feature data statistics:')
        print(iris_df.data.describe())

        print('Distribution by target value:')
        print(iris_df.target.replace({0: 'setosa', 
                                        1: 'versicolor', 
                                        2: 'virginica'}).value_counts())
        ```
        '''
    )
with output_col:
        st.write('#### Output:')
run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))

if run_button: 
    iris_df = load_iris(as_frame=True)
    with output_col:
            st.write('**iris_df.data dataframe:**')
            with st.expander('Expand dataframe'):
                st.write(iris_df.data)
            st.text('\n  ')

            st.write('**iris_df.target series:**')
            with st.expander('Expand series'):
                st.write(iris_df.target)
            st.text('\n  ')
            
            st.write('**Feature data statistics:**')
            with st.expander('Expand statistics'):
                st.write(iris_df.data.describe())
            st.text('\n  ')

            st.write('**Distribution by target value:**')
            with st.expander('Expand Distribution'):
                st.write(iris_df.target.replace({0: 'setosa', 
                                             1: 'versicolor', 
                                             2: 'virginica'}).value_counts())

st.write(
        '''
        From the output of the `.describe()` method on the features dataframe, we see that all of the features fall within a few centimeters of each other. 
        * The sepal length feature has the largest average value while the petal width feature the smallest. 
        * The petal length feature is the most spread out within its range (has the largest standard deviation) 
        *  the sepal width the most constrained (smallest standard deviation).

        From the output of the `.value_counts()` method on the target series (after a replacement of integer values to strings for clarity), we see that there are exactly 50 samples from each species.
        ''')


st.write("\n")
st.write("### Splitting the Data")

st.write(
        '''
        We will make use of the the function `test_train_split()` which can be found in the `sklearn.model_selection` module. Details about this function can be found in its [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). To import it, simply use
        ```python
        from sklearn.model_selection import test_train_split
        ```
        The features are grouped together into one dataset X, and the target in another dataset y. In this case, one can use the simpler syntax
        
        ```python
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        
        ```
        In the call above,  each ```data_train``` and ```data_test``` will contain 80% and 20% of the data respectively. Data in the testing sets should never be used in any part of the training process. 
        The default is to use 25% of the data for testing.
         
        The ```shuffle``` parameter determines whether to shuffle the entries in each dataset prior to splitting.  This is generally a good idea to avoid any bias in how the data was originally ordered. In this case, the iris dataset is ordered by species, and so it must be shuffled before spitting. 
    
        The ```random_state``` parameter can be used to allow results to be replicated across multiple function calls, which is useful when one wants to tune a model without random shuffling affecting the output
        '''
    )

st.write("### Training and testing a classifier")


st.write(
        '''
        First we need to decide which Naive-Bayes classifier is best for our problem. The module ```sklearn.naive_bayes``` includes all of the classifiers covered in the Mathematical Background page: 
        ```MultinomialNB``` , ```ComplementNB```, ```BernoulliNB```, ```CategoricalNB```, and ```GaussianNB```. 
        
        As we've seen above, the iris dataset consists of fully numerical data, and from our exploratory data analysis, each feature is roughly normally distributed in each class.

        This means that Gaussian Naive-Bayes is perfect for this dataset! To use this classifier, (or any of the other variations included in sklearn.naive-bayes), simply import it and instantiate a class instance:
        ```python
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        ```
        
        Since the data is already in a clean and tidy form with no missing values, we can split it into training and testing sets:
        ```python
        iris_df = load_iris(as_frame=True)

        iris_features_train, iris_features_test, iris_species_train, iris_species_test 
                = train_test_split(iris_df.data, iris_df.target, test_size=0.2, shuffle=True, random_state=42)
        ```

        To train our classifier on the training sets `iris_features_train` and `iris_species_train`, we call the `.fit()` method on `classifier`:
        ```python
        classifier.fit(iris_features_train, iris_species_train)
        ```
        After the classifier is trained, its accuracy when applied to the testing set can be found using the `.score()` method:
        ```python
        print(classifier.score(iris_features_test, iris_species_test))
        ```
        
        This will print the _accuracy_ of the classifier on the testing set, which is the number of times the classifier made a correct prediction divided by the total number of predictions made. To see the actual predictions determined from the testing set, we can use the `.predict()` method
        ```python
        iris_species_predict = classifier.predict(iris_species_test)
        ```
        '''
    )


 # ------------------------------------------
    # ----- Classifier training code block -----
    # ------------------------------------------
run_button_key = 'iris_training_run_button'
st.subheader('Code:')
code_col, button_col = st.columns([5,1])
with code_col:
    code_expander = st.expander('Expand code')
    with code_expander:
        st.write(
            '''
            ```python   
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.datasets import load_iris

            # Load the data
            iris_df = load_iris(as_frame=True)
            iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

            # Split the data
            iris_features_train, iris_features_test, iris_species_train, iris_species_test = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)

            # Instantiate a classifier
            classifier = GaussianNB()
            
            # Train the classifier
            classifier.fit(iris_features_train, iris_species_train)

            # Compute the classification score
            print(f'Classifier accuracy: {classifier.score(iris_features_test, iris_species_test)}')

            # Compute predictions for the testing data
            iris_species_predict = classifier.predict(iris_features_test)

            print('Classification report:')
            print(classification_report(iris_species_predict, iris_species_test))

            ```
            ''' 
        )



with button_col:
    run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
st.subheader('Output:')
output_col1, output_col2 = st.columns(2)

if run_button:
    iris_df = load_iris(as_frame=True)
    iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)
    
    iris_features_train, iris_features_test, iris_species_train, iris_species_test \
        = train_test_split(iris_df.data, iris_df.target, train_size=0.2, shuffle=True, random_state=42)

    
    classifier = GaussianNB()
    
    classifier.fit(iris_features_train, iris_species_train)
    iris_species_predict = classifier.predict(iris_features_test)    
    
    
    
    classifier_ber = BernoulliNB()
    classifier_ber.fit(iris_features_train, iris_species_train)
    y_pred = classifier_ber.predict(iris_features_test)
    
    classifier_mul = MultinomialNB()
    classifier_mul.fit(iris_features_train, iris_species_train)
    
    y_pred = classifier_mul.predict(iris_features_test)
   
    with output_col1:
        st.write('**Classifier accuracy:**',classifier.score(iris_features_test, iris_species_test))
        st.markdown("---")
        st.write("accuracy_score of BernoulliNB",classifier_ber.score(iris_features_test, iris_species_test))
        st.markdown("---")
        st.write("accuracy_score of MultinomialNB",accuracy_score(y_pred,iris_species_test))
        
    with output_col2:              
        st.write('**Classification report:**')
        st.text(classification_report(iris_species_predict, iris_species_test))
        