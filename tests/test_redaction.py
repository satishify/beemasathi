from observability.trace import redact


def test_redacts_contact_and_ids():
    text = "mail me at ravi.kumar@example.com or 9876543210, Aadhaar 1234 5678 9012, PAN ABCDE1234F"
    cleaned = redact(text)
    assert "ravi.kumar@example.com" not in cleaned
    assert "9876543210" not in cleaned
    assert "1234 5678 9012" not in cleaned
    assert "ABCDE1234F" not in cleaned
    assert "[email]" in cleaned


def test_redacts_secrets():
    assert "sk-" not in redact("key sk-abcdefghijklmnop1234")
    assert "[secret]" in redact("Authorization: Bearer abcdef123456")
