from eigen3.erl.cloud_sync import is_forbidden_cloud_upload_local_path


def test_forbidden_paths_training_state():
    assert is_forbidden_cloud_upload_local_path("/tmp/training_state.pkl")
    assert is_forbidden_cloud_upload_local_path("training_state.pkl")
    assert is_forbidden_cloud_upload_local_path("foo/training_state.pkl.tmp")


def test_allowed_paths_hof_agent():
    assert not is_forbidden_cloud_upload_local_path(
        "/checkpoints/run/hall_of_fame/hof_agent_0.msgpack"
    )
    assert not is_forbidden_cloud_upload_local_path(
        "/checkpoints/run/best_agent.msgpack"
    )
