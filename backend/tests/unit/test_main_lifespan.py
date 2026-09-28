from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock


def test_initialize_runtime_seeds_declared_social_sources(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import main as module
    from app.database import Base
    from app.infra.db.models.social_signals import SocialSourceConfiguration
    from app.services.social_source_admin_service import SEED_SOCIAL_SOURCES

    engine = create_engine(f"sqlite:///{tmp_path / 'startup.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(module, "migrate_database_to_head", lambda selected: "current")

    module.initialize_runtime()

    with sessions() as db:
        rows = db.query(SocialSourceConfiguration).order_by(
            SocialSourceConfiguration.x_list_id
        ).all()
        assert [(row.x_list_id, row.lifecycle_state, row.provenance) for row in rows] == sorted(
            (list_id, "enabled", "system_seed")
            for list_id, _name, _url in SEED_SOCIAL_SOURCES
        )
    engine.dispose()


def test_group_history_startup_trigger_uses_shutdown_independent_daemon(monkeypatch):
    from app import main as module

    publisher = Mock(name="publisher-thread")
    thread_factory = Mock(return_value=publisher)
    monkeypatch.setattr(module.threading, "Thread", thread_factory)

    result = module.trigger_group_history_reconciliation_on_startup()

    assert result == {"status": "dispatching"}
    thread_factory.assert_called_once_with(
        target=module._publish_startup_work,
        name="startup-publisher",
        daemon=True,
    )
    publisher.start.assert_called_once_with()


def test_daemon_publisher_can_remain_blocked_without_lifespan_owning_it(monkeypatch):
    from app import main as module

    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    discovery = Mock()
    discovery.delay.side_effect = lambda: (
        dispatch_started.set(),
        release_dispatch.wait(),
        SimpleNamespace(id="discovery-1"),
    )[-1]
    monkeypatch.setattr(
        "app.tasks.group_history_tasks.discover_group_history_reconciliation",
        discovery,
    )

    monkeypatch.setattr(module, "initialize_runtime", Mock())
    monkeypatch.setattr(
        module,
        "initialize_process_runtime_services",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(module, "clear_runtime_services", Mock())
    monkeypatch.setattr(module.settings, "mcp_http_enabled", False)
    monkeypatch.setattr(module.engine, "dispose", Mock())
    test_app = SimpleNamespace(state=SimpleNamespace())

    async def run_lifespan():
        async with module.lifespan(test_app):
            assert dispatch_started.wait(timeout=0.1)

    try:
        asyncio.run(asyncio.wait_for(run_lifespan(), timeout=0.2))
    finally:
        release_dispatch.set()


def _sqlite_sessions(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import main as module
    from app.database import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'startup.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(module, "migrate_database_to_head", lambda selected: "current")
    return sessions


def test_initialize_runtime_mirrors_social_deployment_settings(monkeypatch, tmp_path):
    from app import main as module
    from app.services.social_source_admin_service import SocialSourceAdminService

    sessions = _sqlite_sessions(monkeypatch, tmp_path)
    monkeypatch.setattr(module.settings, "social_signals_mode", "live")
    monkeypatch.setattr(module.settings, "social_ingest_provider", "xui")

    module.initialize_runtime()

    with sessions() as db:
        runtime = SocialSourceAdminService(db).read_runtime()
    assert (runtime.mode, runtime.provider) == ("live", "xui")


def test_missing_feature_refreshes_are_queued_once(monkeypatch, tmp_path):
    from app import main as module

    _sqlite_sessions(monkeypatch, tmp_path)
    monkeypatch.setattr(module.settings, "social_signals_mode", "live")
    monkeypatch.setattr(module.settings, "social_ingest_provider", "xui")
    module.initialize_runtime()

    guard_keys = set()
    redis = Mock()
    redis.set.side_effect = lambda key, value, nx, ex: (
        key not in guard_keys and not guard_keys.add(key)
    )
    monkeypatch.setattr("app.services.redis_pool.get_redis_client", lambda: redis)
    cot = Mock()
    social = Mock()
    monkeypatch.setattr("app.interfaces.tasks.cot_tasks.refresh_cot", cot)
    monkeypatch.setattr(
        "app.interfaces.tasks.social_signal_tasks.refresh_social_signals", social
    )

    module._publish_missing_feature_refreshes()
    module._publish_missing_feature_refreshes()  # another worker / restart

    cot.delay.assert_called_once_with(origin="startup")
    social.delay.assert_called_once_with(origin="startup")


def test_failed_feature_enqueue_releases_its_guard_and_continues(monkeypatch, tmp_path):
    from app import main as module

    _sqlite_sessions(monkeypatch, tmp_path)
    monkeypatch.setattr(module.settings, "social_signals_mode", "live")
    monkeypatch.setattr(module.settings, "social_ingest_provider", "xui")
    module.initialize_runtime()

    guard_keys = set()
    redis = Mock()
    redis.set.side_effect = lambda key, value, nx, ex: (
        key not in guard_keys and not guard_keys.add(key)
    )
    redis.delete.side_effect = guard_keys.discard
    monkeypatch.setattr("app.services.redis_pool.get_redis_client", lambda: redis)
    cot = Mock()
    cot.delay.side_effect = ConnectionError("broker unavailable")
    social = Mock()
    monkeypatch.setattr("app.interfaces.tasks.cot_tasks.refresh_cot", cot)
    monkeypatch.setattr(
        "app.interfaces.tasks.social_signal_tasks.refresh_social_signals", social
    )

    module._publish_missing_feature_refreshes()

    social.delay.assert_called_once_with(origin="startup")
    # The COT guard is released so the next worker or restart retries it.
    assert guard_keys == {"startup_refresh:social"}
