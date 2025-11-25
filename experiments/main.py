import neo4j
import logging
import os
import polars as pl

from src.create import Graph
from src.reader import Reader

from utils.dataloader import load_data

logging.basicConfig(level=logging.INFO)

def head(df: pl.DataFrame) -> None:
    """Display the first few rows of the DataFrame."""
    print(df.head())

def delete_all_nodes(neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logging.info("Deleted all nodes in the database.")

def main():

    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        # return count of nodes
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        record = result.single()
        node_count = record["node_count"]
        logging.info(f"Connected to Neo4j database. Node count: {node_count}")

if __name__ == "__main__":
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # g = Graph(neo4j_uri, neo4j_user, neo4j_password)

    with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        with driver.session() as session:
            df = load_data(session)
            print(df)
