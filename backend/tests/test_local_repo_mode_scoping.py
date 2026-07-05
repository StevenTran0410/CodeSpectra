"""Tests for CS-272: AEH must have an independent local_repos lineage from Code
Analysis for the same folder path (own include_tests, own mode-scoped listing).
"""
import pytest

from domain.local_repo.service import LocalRepoService
from domain.local_repo.types import AddLocalRepoRequest
from shared.errors import ConflictError


@pytest.fixture
def service() -> LocalRepoService:
    return LocalRepoService()


async def test_same_path_can_be_added_under_both_modes(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)

    ca_repo = await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    aeh_repo = await service.add(AddLocalRepoRequest(path=path, mode="aeh"))

    assert ca_repo.id != aeh_repo.id
    assert ca_repo.path == aeh_repo.path
    assert ca_repo.mode == "code_analysis"
    assert aeh_repo.mode == "aeh"


async def test_aeh_mode_add_sets_include_tests_true_automatically(
    service: LocalRepoService, tmp_path
) -> None:
    repo = await service.add(AddLocalRepoRequest(path=str(tmp_path), mode="aeh"))
    assert repo.include_tests is True


async def test_code_analysis_mode_add_keeps_include_tests_false(
    service: LocalRepoService, tmp_path
) -> None:
    repo = await service.add(AddLocalRepoRequest(path=str(tmp_path), mode="code_analysis"))
    assert repo.include_tests is False


async def test_duplicate_add_same_path_and_mode_rejected(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)
    await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    with pytest.raises(ConflictError):
        await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))


async def test_list_all_filters_by_mode(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)
    ca_repo = await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    aeh_repo = await service.add(AddLocalRepoRequest(path=path, mode="aeh"))

    ca_only = await service.list_all(mode="code_analysis")
    aeh_only = await service.list_all(mode="aeh")

    ca_ids = {r.id for r in ca_only}
    aeh_ids = {r.id for r in aeh_only}

    assert ca_repo.id in ca_ids
    assert ca_repo.id not in aeh_ids
    assert aeh_repo.id in aeh_ids
    assert aeh_repo.id not in ca_ids
