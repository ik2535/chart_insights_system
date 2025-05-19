#!/usr/bin/env python
"""
Enhanced Neo4j connection test with detailed diagnostics.
"""

import yaml
from neo4j import GraphDatabase
import socket
import subprocess
import sys
import time

def check_port_availability(host, port):
    """Check if a port is open."""
    try:
        socket.create_connection((host, port), timeout=5)
        return True
    except (socket.error, socket.timeout):
        return False

def get_neo4j_status():
    """Check Neo4j process status."""
    try:
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(["sc", "query", "Neo4j"], capture_output=True, text=True)
            if "RUNNING" in result.stdout:
                return "Running (Windows Service)"
            else:
                return "Not running (Windows Service)"
        else:
            # Linux/Mac
            result = subprocess.run(["pgrep", "-f", "neo4j"], capture_output=True, text=True)
            if result.stdout.strip():
                return "Running (Process found)"
            else:
                return "Not running (No process found)"
    except Exception as e:
        return f"Unable to check status: {e}"

def test_neo4j_connection():
    """Test connection to Neo4j database with detailed diagnostics."""
    print("=== Neo4j Connection Diagnostics ===\n")
    
    # Load config
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config file loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        return
    
    # Get Neo4j settings
    neo4j_config = config.get('graph_db', {})
    uri = neo4j_config.get('uri', 'bolt://localhost:7687')
    username = neo4j_config.get('username', 'neo4j')
    password = neo4j_config.get('password', 'password')
    database = neo4j_config.get('database', 'chart_insights')
    
    print(f"Configuration:")
    print(f"  URI: {uri}")
    print(f"  Username: {username}")
    print(f"  Database: {database}")
    print(f"  Password: {'*' * len(password)}")
    print()
    
    # Check Neo4j process status
    print("Checking Neo4j process status...")
    status = get_neo4j_status()
    print(f"  Status: {status}")
    print()
    
    # Extract host and port from URI
    if uri.startswith('bolt://'):
        host_port = uri.replace('bolt://', '')
        if ':' in host_port:
            host, port = host_port.split(':')
            port = int(port)
        else:
            host = host_port
            port = 7687
    else:
        host = 'localhost'
        port = 7687
    
    # Check port availability
    print(f"Checking port {host}:{port}...")
    if check_port_availability(host, port):
        print(f"✅ Port {port} is open and accepting connections")
    else:
        print(f"❌ Port {port} is not accessible")
        print("\nPossible solutions:")
        print("1. Make sure Neo4j is running")
        print("2. Check if Neo4j is running on a different port")
        print("3. Check firewall settings")
        return
    print()
    
    # Attempt connection
    print("Attempting to connect to Neo4j...")
    try:
        # Create driver with explicit timeout
        driver = GraphDatabase.driver(uri, auth=(username, password), connection_timeout=15.0)
        
        print("✅ Driver created successfully")
        
        # Test basic connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            message = result.single()["message"]
            print(f"✅ Basic connection test: {message}")
        
        # Test specific database
        try:
            with driver.session(database=database) as session:
                result = session.run("RETURN 'Database accessible!' as message")
                message = result.single()["message"]
                print(f"✅ Database '{database}' test: {message}")
        except Exception as e:
            print(f"⚠️  Database '{database}' not accessible, but default database works: {e}")
            print("Note: Some Neo4j versions don't support multiple databases")
        
        # Get version info
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions")
            for record in result:
                if record["name"] == "Neo4j Kernel":
                    print(f"Neo4j Version: {record['versions'][0]}")
        
        driver.close()
        print("\n✅ Connection test successful!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        if "ServiceUnavailable" in str(e):
            print("\nTroubleshooting for ServiceUnavailable:")
            print("1. Make sure Neo4j is actually running (check process)")
            print("2. Verify the URI in config.yaml matches your Neo4j setup")
            print("3. Check that Neo4j is listening on the expected port")
            print("4. Try restarting Neo4j")
        elif "AuthError" in str(e):
            print("\nTroubleshooting for Authentication Error:")
            print("1. Check the username and password in config.yaml")
            print("2. Make sure you've set a password for the Neo4j user")
            print("3. Try connecting via Neo4j Browser first")
        else:
            print(f"\nGeneral troubleshooting:")
            print("1. Check Neo4j logs for detailed error messages")
            print("2. Verify network connectivity")
            print("3. Ensure no other application is using port 7687")

if __name__ == "__main__":
    test_neo4j_connection()
