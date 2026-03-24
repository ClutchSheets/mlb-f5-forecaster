import sqlite3

DB_FILE = "mlb_f5_forecasts.db"

def column_exists(conn, table, col):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)

def add_column(conn, table, col, coltype):
    if not column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype};")
        print(f"Added column: {col}")
    else:
        print(f"Column exists: {col}")

def main():
    conn = sqlite3.connect(DB_FILE)

    add_column(conn, "forecast_archive", "away_sp", "TEXT")
    add_column(conn, "forecast_archive", "home_sp", "TEXT")

    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
