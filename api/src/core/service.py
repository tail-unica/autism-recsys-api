from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from neo4j import GraphDatabase
import os

from src.core.utils import get_cfg, get_logger
from src.core.dummy import dummy_place_info_fetcher, dummy_place_recommender
from src.core.info import fetch_place_info

class RecommenderService:
    def __init__(self) -> None:
        self._cfg = None
        self._logger = None
        self._ready = False
        self._warmup_task: Optional[asyncio.Task] = None

        self._model = None
        self._tokenizer = None
        self._indices = None
        self._matcher = None

        self._neo4j_driver = None

    async def initialize(self) -> None:
        self._cfg = get_cfg()
        self._logger = get_logger()

        self._neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
        self._logger.info("Connected to Neo4j database.")

        self._logger.info("RecommenderService: Starting initialization.")

        self._ready = True
        self._logger.info("RecommenderService: Initialization complete.")

    def is_ready(self) -> bool:
        return self._ready

    async def shutdown(self) -> None:
        # TODO: release resources if needed
        self._ready = False
        self._logger and self._logger.info("Core shutdown")

    async def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        #TODO: implement gpu offloading if needed
        if not self._ready:
            raise RuntimeError("Service not ready")

        return dummy_place_recommender(**payload)

    async def fetch_place_info(self, place: str) -> Dict[str, Any]:
        #TODO: implement gpu offloading if needed
        if not self._ready:
            raise RuntimeError("Service not ready")

        with self._neo4j_driver.session() as session:
            info = fetch_place_info(session, place)
        return info