from oscillink.preprocess.autocorrect import smart_correct


def test_basic_typo_corrections_with_case():
    s = "teh quick brown Fox occurence"
    out = smart_correct(s)
    # lowercase/Title case preserved; common typos fixed
    assert out == "the quick brown Fox occurrence"


def test_preserve_acronyms_and_code_like_tokens():
    s = "API teh goverment var_name camelCase HTTP2"
    out = smart_correct(s)
    # 'API' preserved; 'teh' fixed; 'goverment' -> 'government'; snake/camel and digits preserved
    assert out.split(" ")[0] == "API"
    assert " the " in f" {out} "
    assert "government" in out
    assert "var_name" in out
    assert "camelCase" in out
    assert "HTTP2" in out


def test_code_fences_are_respected():
    s = """
This line has teh typo
```
code block with teh and recieve
```
Outside goverment
""".strip()
    out = smart_correct(s)
    # inside code fence should be unchanged; outside should be corrected
    assert "This line has the typo" in out
    assert "code block with teh and recieve" in out
    assert "Outside government" in out


def test_punctuation_boundaries_and_custom_preserve():
    s = "(teh), 'recieve'! And Oscillink is preserved."
    out = smart_correct(s, custom_preserve=["Oscillink"])
    assert out.startswith("(the), 'receive'!")
    assert "Oscillink is preserved." in out


def test_urls_emails_and_non_ascii_are_preserved():
    s = "Visit https://example.com or email foo@bar.com – occurence"
    out = smart_correct(s)
    # url/email intact; non-ascii en dash preserved; typo at end fixed
    assert "https://example.com" in out
    assert "foo@bar.com" in out
    assert "occurrence" in out
