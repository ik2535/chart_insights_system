#!/usr/bin/env python
"""
Initialize Neo4j database for Chart Insights System.
Creates necessary constraints and indexes.
"""

import yaml
from neo4j import GraphDatabase

def initialize_neo4j():
    """Initialize Neo4j database with constraints and indexes."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get Neo4j settings
    neo4j_config = config.get('graph_db', {})
    uri = neo4j_config.get('uri', 'bolt://localhost:7687')
    username = neo4j_config.get('username', 'neo4j')
    password = neo4j_config.get('password', 'password')
    database = neo4j_config.get('database', 'chart_insights')
    
    print(f"Initializing Neo4j database '{database}' at {uri}...")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session(database=database) as session:
            print("Creating constraints and indexes...")
            
            # Create constraints for unique graph_id
            constraints = [
                "CREATE CONSTRAINT chart_id IF NOT EXISTS FOR (c:Chart) REQUIRE c.graph_id IS UNIQUE",
                "CREATE CONSTRAINT point_id IF NOT EXISTS FOR (p:DataPoint) REQUIRE (p.graph_id, p.node_id) IS UNIQUE",
                "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (c:Category) REQUIRE (c.graph_id, c.node_id) IS UNIQUE",
                "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE (s.graph_id, s.node_id) IS UNIQUE",
                "CREATE CONSTRAINT series_id IF NOT EXISTS FOR (s:Series) REQUIRE (s.graph_id, s.node_id) IS UNIQUE",
                "CREATE CONSTRAINT stats_id IF NOT EXISTS FOR (s:Statistics) REQUIRE (s.graph_id, s.node_id) IS UNIQUE",
                "CREATE CONSTRAINT stat_id IF NOT EXISTS FOR (s:Statistic) REQUIRE (s.graph_id, s.node_id) IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✅ Created: {constraint.split()[2]}")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"⚠️  Already exists: {constraint.split()[2]}")
                    else:
                        print(f"❌ Failed: {constraint} - {e}")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX chart_type_idx IF NOT EXISTS FOR (c:Chart) ON (c.chart_type)",
                "CREATE INDEX point_x_idx IF NOT EXISTS FOR (p:DataPoint) ON (p.x)",
                "CREATE INDEX point_y_idx IF NOT EXISTS FOR (p:DataPoint) ON (p.y)",
                "CREATE INDEX category_value_idx IF NOT EXISTS FOR (c:Category) ON (c.value)",
                "CREATE INDEX segment_value_idx IF NOT EXISTS FOR (s:Segment) ON (s.value)",
                "CREATE INDEX stat_name_idx IF NOT EXISTS FOR (s:Statistic) ON (s.name)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    print(f"✅ Created: {index.split()[2]}")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"⚠️  Already exists: {index.split()[2]}")
                    else:
                        print(f"❌ Failed: {index} - {e}")
        
        driver.close()
        print("\n✅ Neo4j database initialization complete!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

if __name__ == "__main__":
    initialize_neo4j()
