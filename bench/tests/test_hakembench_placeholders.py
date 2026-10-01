"""Case suffixes after masking placeholders."""

from bench.hakembench.placeholders import harmonise, suffix, unharmonised


def test_each_case_follows_the_placeholder_word():
    assert [suffix("şirket", c) for c in ("genitive", "dative", "accusative", "locative",
                                           "ablative", "instrumental")] == [
        "in", "e", "i", "te", "ten", "le"]  # fmt: skip
    assert [suffix("ad", c) for c in ("genitive", "dative", "locative", "ablative")] == [
        "ın", "a", "da", "dan"]  # fmt: skip
    assert suffix("ilçe", "genitive") == "nin" and suffix("ilçe", "dative") == "ye"
    assert suffix("kurum", "genitive") == "un" and suffix("ürün", "locative") == "de"
    assert suffix("hesap no", "genitive") == "nun"


def test_harmonise_rewrites_suffixes_left_by_a_mask():
    assert harmonise("[şirket]'nın yeni menüsü") == "[şirket]'in yeni menüsü"
    assert harmonise("[şirket]'nda yemek yedik") == "[şirket]'te yemek yedik"
    assert harmonise("[şirket]'ndaki masa") == "[şirket]'teki masa"
    assert harmonise("[şirket]'na yazdım") == "[şirket]'e yazdım"
    assert harmonise("[şirket]'larına sor") == "[şirket]'lerine sor"
    assert harmonise("[ad]'ın bekası") == "[ad]'ın bekası"
    assert harmonise("[bağlantı]'ya tıklayın") == "[bağlantı]'ya tıklayın"
    assert harmonise("[ad] Hoca geldi") == "[ad] Hoca geldi"


def test_unharmonised_lists_only_what_would_change():
    assert unharmonised("[ad]'dan ve [şirket]'nın") == ["[şirket]'nın"]
    assert unharmonised("temiz metin [ad]'a") == []
