import sqlite3
DB_PATH = "mlb_f5_edge_v2.db"
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM edge_clv_bets")
conn.execute("DELETE FROM edge_clv_summary")
conn.commit()
conn.close()
print("Cleared edge_clv_bets and edge_clv_summary.")
