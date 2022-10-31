# BT4103 Sensemaking of Text Data for Pre-processing   

## Prerequisites

- Python (3.10)

## Setting up

Install dependencies by executing the command below on the terminal

``` cli
pip3 install -r requirements.txt
```

To run the application, execute the command below
Ensure that you are in the main directory before executing the command

``` cli
streamlit run main.py
```

## File Structure

```ml
.
├─ .streamlit
    ├─ config.toml ─ "Theme for frontend"
├─ similarity_metric (Backend)
├─ topic_models (Backend)
    ├─ data.py ─ "Data Loading, Preprocessing and Helper Functions"
    ├─ bertopic.py ─ "Bertopic Model Functions"
    ├─ lda.py ─ "LDA Model Functions"
    ├─ nmf.py ─ "NMF Model Functions"
    ├─ top2vec.py ─ "Top2Vec Model Functions"
├─ main.py (Frontend) ─ "Main Streamlit Script"
├─ requirements.txt ─ "Python Dependencies"
```

## Notes

- Uploading small datasets may result in undesired behaviour or errors for the topic models
