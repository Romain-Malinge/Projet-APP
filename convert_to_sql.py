#!/usr/bin/env python3
import os
import csv
import sqlite3
import argparse

def infer_type(value):
    """Infer SQLite column type from string contents."""
    if value is None or value == "":
        return "TEXT"
    # Integer
    try:
        int(value)
        return "INTEGER"
    except ValueError:
        pass
    # Float
    try:
        float(value)
        return "REAL"
    except ValueError:
        pass
    return "TEXT"


def create_table_from_csv(cursor, table_name, header, sample_row):
    """Create a SQL table using the header and inferred types."""
    column_types = []
    for col_name, sample_value in zip(header, sample_row):
        col_type = infer_type(sample_value)
        column_types.append((col_name, col_type))

    columns_sql = ", ".join([f'"{name}" {ctype}' for name, ctype in column_types])
    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cursor.execute(f'CREATE TABLE "{table_name}" ({columns_sql});')


def insert_csv_into_table(cursor, table_name, header, rows):
    """Insert all rows into the table efficiently."""
    placeholders = ", ".join(["?"] * len(header))
    cursor.executemany(
        f'INSERT INTO "{table_name}" VALUES ({placeholders})', rows
    )


def csv_to_sqlite(csv_folder, sqlite_path):
    """Load all CSV files in a folder into a SQLite database."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    for filename in os.listdir(csv_folder):
        if not filename.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(csv_folder, filename)
        table_name = os.path.splitext(filename)[0]  # filename without extension

        print(f"Importing {filename} -> table '{table_name}'")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        if len(rows) == 0:
            print(f"Skipping empty CSV: {filename}")
            continue

        # Use first row to infer types
        create_table_from_csv(cursor, table_name, header, rows[0])

        insert_csv_into_table(cursor, table_name, header, rows)

    conn.commit()
    conn.close()
    print(f"\nDone! SQLite DB created at: {sqlite_path}")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Convert CSVs to SQLite DB.")
    #parser.add_argument("csv_folder", help="Folder containing CSV files")
    #parser.add_argument("sqlite_path", help="Output SQLite database path")
    #args = parser.parse_args()

    csv_folder = "./AcquisitionsEyeTracker/sujet1_f-42e0d11a"
    sqlite_path = "database.sqlite"

    csv_to_sqlite(csv_folder, sqlite_path)
