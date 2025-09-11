# wrappers/linkedin_wrapper.py
import io
import sys
import os
import importlib.util
import pandas as pd

AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'linkedin_agent'))

def _load_module_from_path(path: str, module_name: str, pkg_path: str | None = None):
    """Load a Python module dynamically from file path."""
    if pkg_path and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_linkedin_agent(df: pd.DataFrame, fail_gracefully: bool = True) -> pd.DataFrame:
    """Enrich dataframe with LinkedIn links via linkedin_agent.file_handler."""
    try:
        fh_path = os.path.join(AGENT_DIR, 'file_handler.py')
        mod = _load_module_from_path(fh_path, 'linkedin_file_handler', pkg_path=AGENT_DIR)
        if not hasattr(mod, "process_csv"):
            raise AttributeError("linkedin_agent.file_handler must define a function `process_csv(bytes_blob)`")

        csv_bytes = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
        print("[linkedin_wrapper] Running LinkedIn agent...")
        processed_bytes = mod.process_csv(csv_bytes)
        print("[linkedin_wrapper] LinkedIn agent finished.")
        out_df = pd.read_csv(io.BytesIO(processed_bytes), encoding='utf-8', dtype=str, keep_default_na=False)
        return out_df
    except Exception as e:
        print(f"[linkedin_wrapper] Error running LinkedIn agent: {e}")
        if fail_gracefully:
            # return original DF so pipeline continues
            return df
        else:
            raise

if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame([{
        'First Name': 'John',
        'Last Name': 'Doe',
        'Company Name': 'Acme',
        'Title': 'CEO',
        'Domain': 'acme.com'
    }])
    print(run_linkedin_agent(df))
