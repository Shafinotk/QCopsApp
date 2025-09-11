# agents/linkedin_agent/main.py
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from file_handler import process_csv  # synchronous function now
import chardet  # for encoding detection

app = FastAPI()

# ---------- Function-based entry for in-memory DataFrame ----------
def run_linkedin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a DataFrame with LinkedIn data.
    Returns a new DataFrame with additional columns.
    """
    # Convert DataFrame to CSV bytes
    csv_bytes = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
    # Process CSV bytes via synchronous logic
    processed_bytes = process_csv(csv_bytes)
    # Return DataFrame from processed CSV bytes
    return pd.read_csv(io.BytesIO(processed_bytes), encoding='utf-8')


# ---------- FastAPI endpoint for file uploads ----------
@app.post("/upload-csv/")
def upload_csv(file: UploadFile):
    # Read uploaded file into memory
    contents = file.file.read()

    # ---------------- Detect encoding ----------------
    detected = chardet.detect(contents[:100000])  # analyze first 100 KB
    encoding = detected["encoding"] or "utf-8"

    # Convert bytes to DataFrame
    df = pd.read_csv(io.BytesIO(contents), encoding=encoding, dtype=str, keep_default_na=False)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Process via synchronous function
    enriched_df = run_linkedin(df)

    # Convert back to CSV bytes
    output_bytes = enriched_df.to_csv(index=False, encoding='utf-8').encode('utf-8')

    # Return downloadable CSV file
    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=processed_{file.filename}"
        },
    )


# ---------- CLI / test entry ----------
if __name__ == "__main__":
    import pandas as pd

    # Quick local test
    df = pd.DataFrame([{
        'First Name': 'John',
        'Last Name': 'Doe',
        'Company Name': 'Acme',
        'Title': 'CEO',
        'Domain': 'acme.com'
    }])

    # Run synchronous function
    enriched_df = run_linkedin(df)
    print(enriched_df)
