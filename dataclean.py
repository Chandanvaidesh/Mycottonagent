from pathlib import Path
from tabulate import tabulate
import pandas as pd
import os
from pypdf import PdfReader
import re
from docling.document_converter import DocumentConverter

p=Path('pest_dataset')
files = p.rglob('*')
pdf_files=[]
csv_files=[]

output_folder="Cleaned_pest"
os.makedirs(output_folder,exist_ok=True)

for f in files:
    if f.suffix=='.pdf':
        pdf_files.append(f)
    elif f.suffix=='.csv':
        csv_files.append(f)
    
for file in csv_files:
    df = pd.read_csv(file)
    print(df.head())
    print(f"shape:{df.shape} column:{df.columns} types : {df.dtypes} missing values: {df.isnull().sum()} stats:{df.describe()} ")     
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.drop_duplicates(inplace=True)     
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)   
    output_path=Path(output_folder) / file.name
    df.to_csv(output_path,index=False)
converter = DocumentConverter()
for file in pdf_files:
    
    result = converter.convert(file)
    doc= result.document
    output_path= Path(output_folder) / (file.name + ".json")

    doc.save_as_json(output_path)