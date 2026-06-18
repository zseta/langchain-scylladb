"""Session-scoped testcontainers fixture for ScyllaDB + vector-store.

Starts two containers on a shared Docker network:
  - scylladb/scylla:2026.1.4   → CQL data node (alias: scylla)
  - scylladb/vector-store:1.7.0 → ANN indexing service (alias: vector-store)

Connection details are yielded as a dict and consumed by the ``vectorstore``
fixture in test_vectorstores.py.

A second ``cloud_session`` fixture connects to ScyllaDB Cloud for live tests
(marked with ``@pytest.mark.cloud``).
"""
from __future__ import annotations

import time
from typing import Generator

import pytest
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT, Session
from cassandra.policies import DCAwareRoundRobinPolicy
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.waiting_utils import wait_for_logs

_SCYLLA_IMAGE = "scylladb/scylla:2026.1.4"
_VECTOR_STORE_IMAGE = "scylladb/vector-store:1.7.0"
_KEYSPACE = "langchain_test"
_LOCAL_DC = "datacenter1"

# ---------------------------------------------------------------------------
# ScyllaDB Cloud credentials (AWS US-East-1 test cluster)
# ---------------------------------------------------------------------------
_CLOUD_HOST = "node-0.aws-us-east-1.76eaa4d045a78a4e4a33.clusters.scylla.cloud"
_CLOUD_USERNAME = "scylla"
_CLOUD_PASSWORD = "H5aT6BIPZ3zlmAU"
_CLOUD_LOCAL_DC = "AWS_US_EAST_1"
_CLOUD_KEYSPACE = "langchain_test"


def pytest_configure(config):
    config.addinivalue_line("markers", "cloud: mark test as requiring ScyllaDB Cloud")


@pytest.fixture(scope="session")
def scylladb_service():
    """Start ScyllaDB + vector-store containers and yield connection info."""
    network = Network()
    network.create()

    scylla = (
        DockerContainer(_SCYLLA_IMAGE)
        .with_network(network)
        .with_network_aliases("scylla")
        .with_command(
            "--smp 1 --memory 1G --overprovisioned 1 "
            "--vector-store-primary-uri http://vector-store:6080 "
            "--broadcast-rpc-address 127.0.0.1"
        )
        .with_exposed_ports(9042)
    )

    vector_store = (
        DockerContainer(_VECTOR_STORE_IMAGE)
        .with_network(network)
        .with_network_aliases("vector-store")
        .with_env("VECTOR_STORE_URI", "0.0.0.0:6080")
        .with_env("VECTOR_STORE_SCYLLADB_URI", "scylla:9042")
    )

    scylla.start()
    wait_for_logs(scylla, "Starting listening for CQL clients", timeout=120)
    time.sleep(5)  # let CQL port fully settle

    vector_store.start()
    wait_for_logs(vector_store, "6080", timeout=60)

    host = "127.0.0.1"
    port = int(scylla.get_exposed_port(9042))

    # Create keyspace via driver (replication_factor=1 for single-node local)
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(_LOCAL_DC)
    )
    cluster = Cluster(
        [host],
        port=port,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        protocol_version=4,
    )
    session = cluster.connect()
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {_KEYSPACE}
        WITH replication = {{'class': 'NetworkTopologyStrategy', '{_LOCAL_DC}': 1}}
        AND tablets = {{'enabled': true}}
        """
    )
    cluster.shutdown()

    yield {"host": host, "port": port, "local_dc": _LOCAL_DC, "keyspace": _KEYSPACE}

    vector_store.stop()
    scylla.stop()
    network.remove()


@pytest.fixture(scope="session")
def cloud_session() -> Generator[Session, None, None]:
    """Open a session to ScyllaDB Cloud (AWS US-East-1 test cluster)."""
    auth = PlainTextAuthProvider(username=_CLOUD_USERNAME, password=_CLOUD_PASSWORD)
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(_CLOUD_LOCAL_DC)
    )
    cluster = Cluster(
        contact_points=[_CLOUD_HOST],
        port=9042,
        auth_provider=auth,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        protocol_version=4,
    )
    session = cluster.connect()
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {_CLOUD_KEYSPACE}
        WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 3}}
        AND tablets = {{'enabled': true}}
        """
    )
    yield session
    cluster.shutdown()
