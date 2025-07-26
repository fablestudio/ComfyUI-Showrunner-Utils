import os
import psycopg2
from typing import Any, Dict

class SR_SupabaseQueryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sql_query": ("STRING", {"multiline": True, "default": "SELECT * FROM your_table LIMIT 10"}),
                "row_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "input_1": ("STRING", {"default": ""}),
                "input_2": ("STRING", {"default": ""}),
                "input_3": ("STRING", {"default": ""}),
                "input_4": ("STRING", {"default": ""}),
                "input_5": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = tuple([f"Column_{i+1}" for i in range(10)] + ["Row_Count"])
    FUNCTION = "run_query"
    CATEGORY = "Showrunner Nodes"

    def run_query(self, sql_query: str, row_index: int, input_1: str = "", input_2: str = "", input_3: str = "", input_4: str = "", input_5: str = ""):
        # Replace #input_1 ... #input_5 in sql_query with their values
        inputs = [input_1, input_2, input_3, input_4, input_5]
        for i, val in enumerate(inputs, 1):
            sql_query = sql_query.replace(f"#input_{i}", val)

        # Get Supabase connection info from environment variables
        host = os.environ.get("SUPABASE_HOST")
        dbname = os.environ.get("SUPABASE_DBNAME")
        user = os.environ.get("SUPABASE_USER")
        password = os.environ.get("SUPABASE_PASSWORD")
        port = os.environ.get("SUPABASE_PORT", 5432)

        if not all([host, dbname, user, password]):
            raise ValueError("Supabase connection info missing in environment variables.")

        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port,
            sslmode="require"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                rows = cur.fetchall()
                row_count = len(rows)
                if row_count == 0:
                    result_row = ["" for _ in range(10)]
                else:
                    idx = min(max(row_index, 0), row_count - 1)
                    result_row = list(rows[idx])
                    # Pad or trim to 10 columns
                    if len(result_row) < 10:
                        result_row += ["" for _ in range(10 - len(result_row))]
                    else:
                        result_row = result_row[:10]
                # Convert all to string for output
                result_row = [str(x) if x is not None else "" for x in result_row]
                return (*result_row, str(row_count))
        finally:
            conn.close()

