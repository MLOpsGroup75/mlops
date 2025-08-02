#!/usr/bin/env python3
"""Script to view all entries from SQLite logs database"""

import sqlite3
import json
import argparse
from datetime import datetime


def view_logs(db_path, limit=None, service=None, level=None, request_id=None):
    """View logs from SQLite database with optional filters"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM logs WHERE 1=1"
        params = []
        
        if service:
            query += " AND service = ?"
            params.append(service)
        
        if level:
            query += " AND level = ?"
            params.append(level.upper())
        
        if request_id:
            query += " AND request_id = ?"
            params.append(request_id)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        print(f"\n📊 Database: {db_path}")
        print(f"📈 Total entries found: {len(rows)}")
        print("=" * 100)
        
        for row in rows:
            id, timestamp, level, logger_name, message, service, request_id, extra_data, created_at = row
            
            print(f"\n🆔 ID: {id}")
            print(f"⏰ Timestamp: {timestamp}")
            print(f"📍 Created: {created_at}")
            print(f"🔢 Level: {level}")
            print(f"🏷️  Logger: {logger_name}")
            print(f"🏢 Service: {service}")
            print(f"🔑 Request ID: {request_id}")
            
            # Try to parse and pretty print JSON message
            try:
                if message and message.startswith('{'):
                    parsed_msg = json.loads(message)
                    print(f"💬 Message: {json.dumps(parsed_msg, indent=2)}")
                else:
                    print(f"💬 Message: {message}")
            except json.JSONDecodeError:
                print(f"💬 Message: {message}")
            
            if extra_data:
                print(f"📋 Extra Data: {extra_data}")
            
            print("-" * 80)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except FileNotFoundError:
        print(f"❌ Database file not found: {db_path}")


def show_summary(db_path):
    """Show summary statistics of the logs"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n📊 Summary for: {db_path}")
        print("=" * 50)
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM logs")
        total = cursor.fetchone()[0]
        print(f"📈 Total entries: {total}")
        
        # Count by level
        cursor.execute("SELECT level, COUNT(*) FROM logs GROUP BY level ORDER BY COUNT(*) DESC")
        levels = cursor.fetchall()
        print(f"\n📊 By Log Level:")
        for level, count in levels:
            print(f"  {level}: {count}")
        
        # Count by service
        cursor.execute("SELECT service, COUNT(*) FROM logs GROUP BY service ORDER BY COUNT(*) DESC")
        services = cursor.fetchall()
        print(f"\n🏢 By Service:")
        for service, count in services:
            print(f"  {service or 'unknown'}: {count}")
        
        # Count by logger
        cursor.execute("SELECT logger_name, COUNT(*) FROM logs GROUP BY logger_name ORDER BY COUNT(*) DESC")
        loggers = cursor.fetchall()
        print(f"\n🏷️  By Logger:")
        for logger, count in loggers:
            print(f"  {logger or 'unknown'}: {count}")
        
        # Recent entries
        cursor.execute("SELECT created_at FROM logs ORDER BY created_at DESC LIMIT 1")
        latest = cursor.fetchone()
        cursor.execute("SELECT created_at FROM logs ORDER BY created_at ASC LIMIT 1")
        earliest = cursor.fetchone()
        
        print(f"\n⏰ Time Range:")
        print(f"  Earliest: {earliest[0] if earliest else 'N/A'}")
        print(f"  Latest: {latest[0] if latest else 'N/A'}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View MLOps logs from SQLite database")
    parser.add_argument("--db", choices=["api", "predict", "both"], default="both",
                        help="Which database to query (default: both)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Limit number of entries to show (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Show all entries (no limit)")
    parser.add_argument("--service", type=str,
                        help="Filter by service name")
    parser.add_argument("--level", type=str,
                        help="Filter by log level (INFO, ERROR, WARNING, etc.)")
    parser.add_argument("--request-id", type=str,
                        help="Filter by specific request ID")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary statistics only")
    
    args = parser.parse_args()
    
    # Database paths
    api_db = "logs/api_logs.db"
    predict_db = "logs/predict_logs.db"
    
    limit = None if args.all else args.limit
    
    if args.summary:
        if args.db in ["api", "both"]:
            show_summary(api_db)
        if args.db in ["predict", "both"]:
            show_summary(predict_db)
    else:
        if args.db in ["api", "both"]:
            view_logs(api_db, limit, args.service, args.level, args.request_id)
        if args.db in ["predict", "both"]:
            view_logs(predict_db, limit, args.service, args.level, args.request_id)