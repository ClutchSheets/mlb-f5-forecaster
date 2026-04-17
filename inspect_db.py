import os
import sqlite3
DB_PATH = "mlb_f5_edge_v2.db"
def main():
    abs_path = os.path.abspath(DB_PATH)
    print(f"Using DB file: {abs_path}")
    print(f"Exists: {os.path.exists(DB_PATH)}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    print("\nTables found:")
    for (table_name,) in tables:
        print(f"- {table_name}")
    for (table_name,) in tables:
        print(f"\nSchema for {table_name}:")
        cols = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
        for col in cols:
            print(col)
        count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Row count: {count}")
    conn.close()
if __name__ == "__main__":
    main()
