import streamlit as st
from PIL import Image

# Import function to fetch dataset
from sklearn.datasets import load_iris
# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import classification_report, confusion_matrix

# Import heatmap plotting function
#from matrix_heatmap import matrix_heatmap

# Import custom latex display and numbering class
#from latex_equation_numbering import latex_equation_numbering

# state = _get_state()

st.page_config = st.set_page_config(
    page_title="Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

        The function call has the general form:
        ```python
        dataset_1_train, dataset_1_test, dataset_2_train, dataset_2_test, ... , dataset_N_train, dataset_N_test 
                = train_test_split(dataset_1, dataset_2, ... , dataset_N, train_size = 0.8, test_size=0.2, shuffle=True, random_state=42, stratify=dataset_j)
        ```
        Each positional arguement (the `dataset_i`'s in the function call) is a dataset to be split. The lengths of each `dataset_i` must be equal. For each one, there is a return value of `dataset_i_train` and `dataset_i_test`. The order of the arguments matches the order of the return pairs. The `train_size` and `test_size` values is the percentage of data to leave for training and testing . In the call above, each `dataset_i_train` and `dataset_i_test` will contain 80% and 20% of the data respectively. Data in the testing sets should _never_ be used in any part of the training process. The example above uses both `train_size` and `test_size` for illustration, but only one is necessary. If either is excluded, the other will be the complement of the one provided. If neither are included, the default is to use 25% of the data for testing. The `shuffle` parameter determines whether to shuffle the entries in each dataset prior to splitting. This is generally a good idea to avoid any bias in how the data was originally ordered. In our case, the iris dataset is ordered by species, and so it must be shuffled before spitting. The `random_state` parameter can be used to allow results to be replicated across multiple function calls, which is useful when one wants to tune a model without random shuffling affecting the output. Finally, `stratify` can be used with labeled data if the target feature (here `dataset_j` above) is unbalanced, meaning the distribution of classes isn't uniform. If we have `stratify=True`, then stratified sampling is used to ensure that the resulting split datasets, `dataset_i_train` and `dataset_i_test`, for each `dataset_i` will have the same proportion of classes as that in `dataset_j`. 
        
        The form of the function call above is for when all of the features and target are in different datasets. Usually, this is not the case, and the features are grouped together into one dataset `X`, and the target in another dataset `y`. In this case, one can use the simpler syntax
        ```python
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
        ```
        '''
    )

st.write("### Training and testing a classifier")


st.write(
        '''
        Finally we are ready to start classification! First we need to decide which Naive-Bayes classifier is best for our problem. The module `sklearn.naive_bayes` includes all of the classifiers covered in the **Mathematical Background** page: `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB`, and `GaussianNB`. As we've seen above, the iris dataset consists of fully numerical data, and from our exploratory data analysis, each feature is roughly normally distributed in each class (determined by looking at the shape of the violins in the species-divided violin plot). This means that Gaussian Naive-Bayes is perfect for this dataset! To use this classifier, (or any of the other variations included in `sklearn.naive-bayes`), simply import it and instantiate a class instance:
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
        We don't need stratified sampling since the iris dataset is balanced (fifty samples for each species), but we do need to apply shuffling before splitting.

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
        The variable `iris_species_predict` contains a list of species predictions for each sample in the testing set. We can use `sklearn`'s built in metrics to evaluate the performance of the classifer. Useful functions in the `sklearn.metrics` module include `classification_report` and `confusion_matrix`:
        ```python
        from sklearn.metrics import classification_report, confusion_matrix
        ```
        These functions and their outputs is covered in the **Mathematical Background** page. The output of `classification_report` is a listing of several statistics that give an overview of the performance of the classifier. The output of `confusion_matrix` is a square matrix with entries corresponding to the types of predictions made (true positives, false positives, true negatives, and false negatives). The classification report can be directly printed, or outputed as a key-value dictionary for easy access to each statistic's value. The confusion matrix is best viewed as a heatmap, similar to the one made for the correlation matrix. The confusion matrix should have large values on the main diagonal and small values elsewhere. The confusion matrix is also non-negative, meaning each value is zero or greater. When the confusion matrix is unnormalized, each value in row A column B corresponds to the number of predictions of class B for samples with a ground truth of class A. The matrix can be row (column) normalized, in which each element is divided by the sum of the elements in each row (column), or population normalized, where each element is divided by the number of predictions made. When the matrix is row normalized, the main diagonal entries correspond to the recall value for each class. When the matrix is column normalized, the main diagonal contains the precision of each class. When the matrix is normalized by population, the sum of the diagonal terms correspond to the model accuracy. In any case, a well-performing model has large values on the main diagonal and small values elsewhere.

        Run the code block to split the iris dataset, train a Gaussian Naive-Bayes classifier on the training portion, and print the model accuracy, classification report, and unnormalized confusion matrix heatmap. 
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

            # Create confusion matrix DataFrame
            cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test), columns=iris_df.target_names, index=iris_df.target_names)

            print('Confusion matrix:')
            print(cm_df)

            # Create a heatmap of the confusion matrix
            fig, ax = plt.subplots()
            ax = sns.heatmap(cm_df.values.tolist(), annot=True, fmt='0.3g', cmap='bone')
            ax.set_title('Confusion matrix heatmap')
            ax.set_xlabel('Species')
            ax.set_ylabel('Species')
            fig.show()
            ```
            '''
        )


# with button_col:
#     run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
# st.subheader('Output:')
# output_col1, output_col2 = st.beta_columns(2)

# if run_button or st.session_state[run_button_key+'_dict']['was_pressed']:
#     st.session_state[run_button_key+'_dict']['was_pressed'] = True

#     iris_df = load_iris(as_frame=True)
#     iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

#     iris_features_train, iris_features_test, iris_species_train, iris_species_test \
#         = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)
    
#     classifier = GaussianNB()

#     classifier.fit(iris_features_train, iris_species_train)

#     iris_species_predict = classifier.predict(iris_features_test)

#     # Create confusion matrix DataFrame
#     # cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test), columns=iris_df.target_names, index=iris_df.target_names)

#     # # Make a heatmap of the confusion matrix
#     # fig, ax = plt.subplots()
#     # fig = matrix_heatmap(cm_df.values.tolist(), options={'x_labels': iris_df.target_names,'y_labels': iris_df.target_names, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (-1,1), 'center': None, 'title_axis_labels': ('Confusion matrix heatmap', 'Species', 'Species'), 'rotate x_tick_labels': True})

#     # with output_col1:
#     #     st.write(f'**Classifier accuracy:** {classifier.score(iris_features_test, iris_species_test)}')

#     #     st.write('**Classification report:**')
#     #     st.text('.  \n'+classification_report(iris_species_predict, iris_species_test))

#     #     st.write('**Confusion matrix:**')
#     #     st.write(cm_df)

#     # with output_col2:              
#     #     st.pyplot(fig)
st.subheader('')