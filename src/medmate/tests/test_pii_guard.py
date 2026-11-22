from medmate.guards.pii import redact_pii


def test_redacts_common_identifiers_and_contact_details() -> None:
    text = (
        "Patient name: Casey Doe, DOB: 1984-03-17, MRN: MRN-3920, "
        "email: casey@example.com, phone: (212) 555-0184, SSN: 123-45-6789"
    )

    redacted = redact_pii(text)

    assert "Casey Doe" not in redacted
    assert "1984-03-17" not in redacted
    assert "MRN-3920" not in redacted
    assert "casey@example.com" not in redacted
    assert "(212) 555-0184" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_NAME]" in redacted
    assert "[REDACTED_DOB]" in redacted
    assert "[REDACTED_ID]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_SSN]" in redacted


def test_redacts_address_value() -> None:
    assert redact_pii("Address: 12 Oak Street, Springfield, NY 10001") == (
        "Address: [REDACTED_ADDRESS]"
    )


def test_leaves_non_pii_text_unchanged() -> None:
    text = "Patient asks about the timing of their next appointment."
    assert redact_pii(text) == text
