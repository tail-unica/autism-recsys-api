# Data Loading Utilities

## Configuration file
The dataset configuration is defined in the `experiments/config/dataset.yaml` file. This file specifies how to interpret the neo4j knowledge graph data, including the mapping of entity and relation IDs to their corresponding tokens.

- `__file__`: Create a new file from this configuration.
- `__type__`: Specify the type of the configuration element (e.g., GRAPH, NODE, RELATION, CSV).
- `__origin__`: Indicates how to connect the keys to the source (e.g., link to the knowledge graph).
- `__from__`: Specifies the source of the data (e.g., kg for knowledge graph).
- `__self__`: Defines the entity id for nodes or relations.
- `__yield__`: Defines the attributes to be extracted for nodes or relations.