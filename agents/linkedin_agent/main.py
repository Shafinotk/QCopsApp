# agents/linkedin_agent/main.py
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from file_handler import process_csv
import chardet

app = FastAPI()

# ---------- Function-based entry ----------
def run_linkedin(df: pd.DataFrame) -> pd.DataFrame:
    csv_bytes = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
    processed_bytes = process_csv(csv_bytes)
    return pd.read_csv(io.BytesIO(processed_bytes), encoding='utf-8')

@app.post("/upload-csv/")
def upload_csv(file: UploadFile):
    contents = file.file.read()
    detected = chardet.detect(contents[:100000])
    encoding = detected["encoding"] or "utf-8"

    df = pd.read_csv(io.BytesIO(contents), encoding=encoding, dtype=str, keep_default_na=False)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    enriched_df = run_linkedin(df)

    output_bytes = enriched_df.to_csv(index=False, encoding='utf-8').encode('utf-8')
    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )

if __name__ == "__main__":
    df = pd.DataFrame([{
        'First Name': 'John',
        'Last Name': 'Doe',
        'Company Name': 'Acme',
        'Title': 'CEO',
        'Domain': 'acme.com',
        'Email': 'john.doe@acme.com'
    }])
    enriched_df = run_linkedin(df)
    print(enriched_df)
