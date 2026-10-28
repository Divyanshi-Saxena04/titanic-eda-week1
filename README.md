# Titanic Data Preparation & EDA (Week 1)

Data cleaning, preprocessing and exploratory data analysis of the public Titanic passenger dataset using Python (Pandas, NumPy, Matplotlib, Seaborn).

## Files
- `titanic.csv` - raw data (source: https://github.com/datasciencedojo/datasets)
- `titanic_clean.csv` - cleaned data with engineered features
- `analysis.py` - full cleaning + EDA script
- `charts/` - six visualizations
- `Week1_Data_Preparation_EDA_Report.docx` - full written report

## Run
```
pip install pandas numpy matplotlib seaborn
mkdir figs
python analysis.py
```

## Key findings
- Women survived at about 74% vs 19% for men.
- Class strongly affected survival (63% / 47% / 24%).
- Children survived most often; families of 2-4 did best.
