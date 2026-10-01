"""r5's overlap check: a train text near a dev text is found in both directions."""

from data.label.r5 import pair_touches

LONG = ("Müşteri hizmetleri asistanı olarak siparişimin kargo durumunu ve teslim tarihini "
        "kontrol edip bana kısa bir özet yazar mısın lütfen")  # fmt: skip


def test_a_copy_and_a_contained_text_are_caught_and_a_stranger_is_not():
    dev = {"d1": LONG}
    train = {
        "copy": LONG + " teşekkürler",
        "inside": LONG.split(" asistanı ")[1],
        "other": "Bugün hava çok güzel, parkta yürüyüş yapıp kitap okumayı düşünüyorum ama "
        "önce market alışverişini bitirmem gerekiyor sanırım",
    }
    found = pair_touches(dev, train)
    assert "copy" in found and "other" not in found
    assert "covered" in found["inside"]
