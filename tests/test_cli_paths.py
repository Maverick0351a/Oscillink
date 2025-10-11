from oscillink import cli


def test_main_no_subcommand_prints_help_and_exits_zero(capsys):
    # argparse prints help and main returns 1 when no subcommand; assert that behavior
    rc = cli.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Oscillink CLI" in captured.out


def test_portal_not_logged_in_message(capsys, monkeypatch, tmp_path):
    # ensure config is empty by pointing CONFIG_DIR to temp
    monkeypatch.setenv("OSCILLINK_CONFIG_DIR", str(tmp_path))
    rc = cli.main(["portal"])  # without login
    out = capsys.readouterr().out
    assert rc == 1
    assert "Not logged in" in out


def test_login_runtime_error_mapping(monkeypatch, capsys, tmp_path):
    # Force _http_request to raise a URLError-mapped RuntimeError; main should return 2
    monkeypatch.setenv("OSCILLINK_CONFIG_DIR", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("Request failed: <urlopen error>")

    monkeypatch.setattr(cli, "_http_request", boom)
    rc = cli.main(["login", "--code", "deadbeef"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Request failed" in err
