"""Shared inline texts and a row builder for the decontamination tests."""

from schema.questions import NoulQuestion
from schema.rows import TrainingRow

# About a hundred tokens of plain Turkish, long enough for stable MinHash estimates.
PARAGRAPH = (
    "Karadeniz kıyısındaki küçük kasabada sabahlar her zaman aynı sesle başlardı. "
    "Balıkçılar gün doğmadan limana iner, ağlarını kontrol eder ve motorlarını "
    "çalıştırırdı. Kahvehanenin sahibi Rıza amca ocağı yakar, ilk çayı kendisi "
    "içerdi. Okula giden çocuklar yokuşu koşarak çıkar, fırının önünde durup sıcak "
    "simit alırdı. Öğleye doğru pazar yeri kalabalıklaşır, köylerden gelen kadınlar "
    "fındık, mısır ve taze peynir satardı. Akşam olunca rüzgâr denizden esmeye "
    "başlar, sokak lambaları tek tek yanar ve kasaba yavaş yavaş sessizliğe gömülürdü. "
    "Yaşlılar bu düzenin yüz yıldır değişmediğini söyler, gençler ise büyük şehirlere "
    "gitmenin hayalini kurardı."
)

UNRELATED = (
    "Ankara'daki teknoloji fuarında bu yıl yapay zekâ ve robotik alanında yüzlerce "
    "şirket ürünlerini tanıttı. Ziyaretçiler insansı robotların yürüyüşünü izledi, "
    "otonom araçların sürüş testlerine katıldı ve yeni nesil işlemcilerin enerji "
    "verimliliği üzerine yapılan sunumları dinledi. Organizatörler fuarın gelecek "
    "yıl İzmir'de daha geniş bir alanda düzenleneceğini açıkladı. Katılımcı firmalar "
    "arasında üniversitelerden çıkan girişimler de vardı ve bazıları yatırımcılarla "
    "ilk görüşmelerini fuar sırasında yaptı."
)


def turkish_upper(text: str) -> str:
    """str.upper maps "i" to "I", which is Turkish "ı"; Turkish capitals are "İ" and "I"."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def make_row(state: str, instructions: str = "Bu metin olumlu mu?") -> TrainingRow:
    return TrainingRow(
        track="deneme",
        task="deneme",
        split="train",
        origin="authored",
        label_kind="rule",
        source="test",
        state=state,
        question=NoulQuestion(instructions=instructions),
        target={"true": 1.0, "false": 0.0},
        recipe="test",
    )
