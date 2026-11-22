from rag.policy_match import match_policy, normalize


def _match(query, candidates, insurer=None):
    return match_policy(normalize(query).split(), candidates, insurer)


def test_reordered_words_prefer_the_longer_product():
    candidates = [
        ("bajaj_allianz", "health_guard"),
        ("bajaj_allianz", "group_guard"),
        ("bajaj_allianz", "group_health_guard_silver"),
    ]
    assert (
        _match("Bajaj Health Guard Group Silver", candidates, "bajaj_allianz")
        == "group_health_guard_silver"
    )


def test_ingestion_artifacts_are_ignored():
    candidates = [
        ("tata_aig", "tata_aig_health_supercharge_super_charge"),
        ("hdfc_ergo", "my_health_koti_suraksha_myhealth"),
        ("max_bupa", "health_plus_healthplus"),
        ("cholamandalam", "chola_flexi_health_supreme_wef"),
    ]
    assert (
        _match("Tata AIG Health Supercharge", candidates, "tata_aig")
        == "tata_aig_health_supercharge_super_charge"
    )
    assert (
        _match("HDFC Ergo My Health Koti Suraksha", candidates, "hdfc_ergo")
        == "my_health_koti_suraksha_myhealth"
    )
    assert (
        _match("Niva Bupa Health Plus", candidates, "max_bupa")
        == "health_plus_healthplus"
    )
    assert (
        _match("Chola Flexi Health Supreme", candidates, "cholamandalam")
        == "chola_flexi_health_supreme_wef"
    )


def test_split_words_are_joined():
    candidates = [("tata_aig", "tata_aig_criti_medi_care_medicare")]
    assert (
        _match("Tata AIG Criti Medi Care", candidates, "tata_aig")
        == "tata_aig_criti_medi_care_medicare"
    )


def test_partial_name_matches_but_a_different_product_does_not():
    candidates = [("cholamandalam", "flexi_health_g")]
    assert _match("Chola Flexi Health", candidates, "cholamandalam") is None
    assert _match("Chola Flexi Health Supreme", candidates, "cholamandalam") is None

    hdfc = [("hdfc_ergo", "my_health_koti_suraksha_myhealth")]
    assert (
        _match("HDFC Ergo My Health Koti", hdfc, "hdfc_ergo")
        == "my_health_koti_suraksha_myhealth"
    )


def test_generic_word_permutations_do_not_match():
    candidates = [("care_health", "care_health")]
    assert _match("does health care cover maternity", candidates) is None


def test_ambiguous_match_returns_none():
    candidates = [
        ("hdfc_ergo", "health_suraksha"),
        ("icici_lombard", "health_suraksha"),
    ]
    assert _match("What is the room rent cap in Health Suraksha?", candidates) is None
