import streamlit as st
from PIL import Image

# Import function to fetch dataset
from sklearn.datasets import load_iris


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





run_button_key = 'iris_stats_run_button'
code_col, output_col = st.columns(2)
with code_col:
    st.subheader('Code:')
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
        st.subheader('Output:')
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
