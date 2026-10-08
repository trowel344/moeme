from scripts.diagnose_moeme_pipeline import diagnose


def status_through_smoke() -> dict:
    return {
        "free_cloud": {
            "activation_capture": {
                "ready": True,
                "service": {"running": False, "result": "success"},
            },
            "postcapture_supervisor": {"status": "passed", "current_stage": "complete"},
            "upload_stage": {"ready": True},
            "free_tier_budget": {"passed": True, "recommended_provider": "lightning"},
            "local_smoke": {"run": {"status": "passed"}},
            "training": {},
            "acceptance": None,
        }
    }


def test_diagnosis_reports_live_capture_as_authoritative_gate() -> None:
    status = status_through_smoke()
    status["free_cloud"]["activation_capture"] = {
        "ready": False,
        "service": {"running": True, "result": "success"},
        "progress": {"completion_fraction": 0.5},
    }
    status["free_cloud"]["postcapture_supervisor"] = {"status": "waiting"}
    result = diagnose(status)
    assert result["overall"] == "running"
    assert result["current_gate"] == "activation_capture"
    assert "do not start competing GPU work" in result["next_action"]


def test_diagnosis_stops_at_first_failed_postcapture_stage() -> None:
    status = status_through_smoke()
    status["free_cloud"]["postcapture_supervisor"] = {
        "status": "failed",
        "current_stage": "imatrix",
        "error": "bad receipt",
    }
    status["free_cloud"]["upload_stage"] = {"ready": False}
    result = diagnose(status)
    assert result["overall"] == "failed"
    assert result["current_gate"] == "postcapture"
    assert result["stages"][1]["details"]["error"] == "bad receipt"


def test_ready_upload_requests_private_cloud_action() -> None:
    result = diagnose(status_through_smoke())
    assert result["overall"] == "user_action_required"
    assert result["current_gate"] == "cloud_training"
    assert "privately upload" in result["next_action"]
    assert "lightning" in result["next_action"]


def test_receipted_upload_advances_action_to_remote_bootstrap() -> None:
    status = status_through_smoke()
    status["free_cloud"]["upload_transfer"] = {
        "status": "passed",
        "provider": "lightning",
        "destination": "lit:///studios/moeme/layer63-200k",
    }
    result = diagnose(status)
    assert result["current_gate"] == "cloud_training"
    assert "open the receipted private lightning destination" in result["next_action"]
    assert result["stages"][4]["details"]["upload_status"] == "passed"


def test_receipted_kaggle_upload_is_not_mislabeled_as_lightning() -> None:
    status = status_through_smoke()
    status["free_cloud"]["upload_transfer"] = {
        "status": "passed",
        "provider": "kaggle",
        "destination": "owner/moeme-private",
    }
    result = diagnose(status)
    assert "private T4 job" in result["next_action"]
    assert "launch_kaggle_kernel.py" in result["next_action"]


def test_submitted_kaggle_kernel_advances_to_read_only_polling() -> None:
    status = status_through_smoke()
    status["free_cloud"]["kaggle_kernel"] = {
        "status": "submitted",
        "kernel": "owner/moeme-layer63-training",
    }
    result = diagnose(status)
    assert result["overall"] == "running"
    assert "sync_kaggle_kernel.py" in result["next_action"]
    assert result["stages"][4]["details"]["kaggle_kernel_status"] == "submitted"


def test_completed_kaggle_kernel_advances_to_verified_sync() -> None:
    status = status_through_smoke()
    status["free_cloud"]["kaggle_kernel"] = {
        "status": "remote_complete",
        "remote_status": "COMPLETE",
    }
    result = diagnose(status)
    assert result["overall"] == "running"
    assert "--sync" in result["next_action"]


def test_interrupted_cloud_run_prescribes_exact_resume() -> None:
    status = status_through_smoke()
    status["free_cloud"]["training"] = {
        "run": {"status": "interrupted", "stage": "training", "attempt": 2}
    }
    result = diagnose(status)
    assert result["overall"] == "interrupted"
    assert result["current_gate"] == "cloud_training"
    assert "resume" in result["next_action"]


def test_interrupted_smoke_prescribes_supervised_exact_resume() -> None:
    status = status_through_smoke()
    status["free_cloud"]["local_smoke"] = {"run": {"status": "interrupted"}}
    result = diagnose(status)
    assert result["current_gate"] == "local_seeded_smoke"
    assert result["overall"] == "interrupted"
    assert "exact atomic smoke state" in result["next_action"]


def test_scientifically_weak_curve_is_rejected_before_transfer() -> None:
    status = status_through_smoke()
    status["free_cloud"]["training"] = {
        "run": {"status": "passed"},
        "curve_analysis": {
            "status": "weak_or_uncertain",
            "best": {"relative_l2": 0.08},
        },
    }
    result = diagnose(status)
    assert result["overall"] == "rejected"
    assert result["current_gate"] == "learning_curve"
    assert result["stages"][5]["details"]["best"] == 0.08


def test_verified_but_ineligible_transfer_is_scientifically_rejected() -> None:
    status = status_through_smoke()
    status["free_cloud"]["training"] = {
        "run": {"status": "passed"},
        "curve_analysis": {
            "status": "promising",
            "best": {"relative_l2": 0.06},
        },
        "transfer_verification": {
            "candidate_ready": False,
            "scientific_eligibility": {"passed": False},
        },
    }
    result = diagnose(status)
    assert result["current_gate"] == "result_transfer"
    assert result["overall"] == "rejected"
