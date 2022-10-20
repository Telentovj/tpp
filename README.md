# BT4103 Sensemaking of Text Data for Pre-processing   

> Important note: Please create your own branch when working on changes, don't need to do PR but also don't override master

## Setting up

Install dependencies

``` cli
pip3 install -r requirements.txt
```

For running front end

``` cli
streamlit run main.py
```

## File Structure

```ml
.
├─ .streamlit
    ├─ config.toml ─ "Theme for frontend"
├─ topic_models (Backend)
    ├─ data.py ─ "Data Loading, Preprocessing and Helper Functions"
    ├─ bertopic.py ─ "Bertopic Model Functions"
    ├─ lda.py ─ "LDA Model Functions"
    ├─ nmf.py ─ "NMF Model Functions"
    ├─ top2vec.py ─ "Top2Vec Model Functions"
├─ main.py (Frontend) ─ "Main Streamlit Script"
├─ requirements.txt ─ "Python Dependencies"
```
