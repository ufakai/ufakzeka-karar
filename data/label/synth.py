"""Decision rows for the tracks no licensed set covers: spam and phishing, education,
legal-aid routing and check-worthiness.

Runs on the lab's server, where the API key lives:

    python -m data.label.synth --out results/step3/synth --cap-usd 8

The questions are ours, written here in Turkish; they are not generated. Where
no licensed text exists, the generator writes the texts, each of an intended
kind (a phishing message, a student answer of a given quality, a legal problem
for a given court), in batches, with fictional names and brands and every link
or number masked, so nothing written here works as a real message. Where text
exists (FACTurk claims for check-worthiness), it is used as it is. Every pair is
labelled by the two judges, as in the build: the target is their mean
vote, and the intended kind is kept beside it only to measure how often the
judges agree with what the generator meant to write.

Texts are split by a hash of their id, a fifth for validation; FACTurk claims
keep the split their converter gave them, so no claim crosses splits. A
template's most common answer is capped in training as in the build, and the
validation rows keep the real mix. Everything is journaled; a restart on the
same output pays for nothing twice, and the client's cap counts the whole
ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from data.label.build import (
    CAP_TOP_SHARE,
    STOPPING,
    balance,
    is_transient,
    no_fit,
    read_jsonl,
    top_of,
)
from data.label.judge import LowLabelMass, label_row
from data.label.pilot import load_panel, load_relabel_judge, make_client, question_of
from data.label.texts import mask_personal_data
from schema.rows import TrainingRow, mean_vote, outcomes, text_of

RECIPE = "synth-v1"
SOURCE = "generated:synth-v1"
VALIDATION_PERCENT = 20
BATCH = 10

GENERATOR_SYSTEM = (
    "Türkçe eğitim verisi yazan dikkatli bir yazarsın. Yalnızca istenen JSON dizisini yaz. "
    "Gerçek kişi, marka, banka, kurum ya da site adı kullanma; uydurma adlar kullan. "
    "Bağlantıların yerine [bağlantı], telefon numaralarının yerine [telefon] yaz."
)


@dataclass(frozen=True)
class Kind:
    """One intended kind of text, with the scenes that vary what is written."""

    name: str
    brief: str
    scenes: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class Ask:
    """One authored question: its task name, track, the question, and how the
    intended kind maps to the answer it should get (for the agreement measure)."""

    task: str
    track: str
    question: dict
    expected: dict[str, str] = field(default_factory=dict)
    # Kinds whose texts get this question; empty means every kind.
    only: tuple[str, ...] = ()


# spam and phishing --------------------------------------------------------

MESSAGE_KINDS = (
    Kind(
        "islem",
        "bir şirketin müşterisine gönderdiği gerçek bir işlem bildirimi (kargo, "
        "randevu, fatura, doğrulama kodu, sipariş); kişiden şifre ya da para istemez",
        (
            "kargo teslimatı",
            "hastane randevusu",
            "elektrik faturası",
            "tek kullanımlık kod",
            "sipariş onayı",
            "banka hesap hareketi",
            "abonelik yenileme",
            "uçuş bilgisi",
        ),
        600,
    ),
    Kind(
        "pazarlama",
        "müşterisi olunan bir şirketin izinli kampanya mesajı; bazılarında ret seçeneği olsun",
        (
            "market indirimi",
            "giyim kampanyası",
            "operatör paketi",
            "restoran fırsatı",
            "kitap indirimi",
            "sadakat puanı",
            "yeni ürün duyurusu",
        ),
        600,
    ),
    Kind(
        "istenmeyen",
        "hiç ilişki kurulmamış bir yerden gelen istenmeyen reklam; kişiyi "
        "dolandırmaya çalışmaz ama izinsiz ve alakasızdır; bazılarında ret seçeneği olsun",
        (
            "kilo verme ürünü",
            "kredi kartı teklifi",
            "emlak projesi",
            "kurs reklamı",
            "kozmetik",
            "oto yedek parça",
            "tatil paketi",
        ),
        600,
    ),
    Kind(
        "dolandirici",
        "kişiyi para, şifre, doğrulama kodu ya da kişisel bilgi vermeye "
        "kandırmaya çalışan bir dolandırıcılık mesajı",
        (
            "sahte kargo gümrük ücreti",
            "hesabınız askıya alındı",
            "vergi iadesi",
            "büyük ödül kazandınız",
            "sahte iş teklifi",
            "tanıdık gibi para isteme",
            "sahte fatura borcu",
            "kod paylaşma isteği",
        ),
        600,
    ),
    Kind(
        "kisisel",
        "iki kişi arasındaki gündelik kişisel mesaj",
        (
            "aile",
            "arkadaş buluşması",
            "iş arkadaşı",
            "komşu",
            "okul",
            "doğum günü",
            "alışveriş listesi",
        ),
        600,
    ),
)
MESSAGE_TYPE = {
    "type": "choice",
    "instructions": "Bu mesaj ne tür bir mesaj?",
    "criteria": {
        "işlem bildirimi": "Bir şirketin müşterisine yaptığı işlemle ilgili bildirim.",
        "izinli pazarlama": "Müşterisi olunan bir yerden gelen kampanya ya da duyuru.",
        "istenmeyen reklam": "İlişki kurulmamış bir yerden gelen izinsiz reklam.",
        "dolandırıcılık": "Kişiyi para, şifre ya da bilgi vermeye kandırma girişimi.",
        "kişisel yazışma": "İki kişi arasındaki gündelik mesaj.",
    },
}
PHISHING = {
    "type": "noul",
    "instructions": (
        "Bu mesaj, kişiyi para, şifre, doğrulama kodu ya da kişisel bilgi vermeye kandırmaya "
        "çalışan bir dolandırıcılık girişimi mi?"
    ),
    "criteria": {
        "true": "Evet, mesaj bir dolandırıcılık girişimi.",
        "false": "Hayır, mesaj bir dolandırıcılık girişimi değil.",
    },
}

# education ----------------------------------------------------------------

# Distinct lise subjects: the smoke run put the middle-school umbrellas (Fen
# Bilimleri, Sosyal Bilgiler) beside the subjects they contain, and the judges
# agreed with the intended subject on two answers in three.
SUBJECTS = ("Türkçe", "Matematik", "Fizik", "Kimya", "Biyoloji", "Tarih", "Coğrafya", "Felsefe")
GRADE = {
    "type": "score",
    "instructions": "Öğrencinin cevabı soruyu ne ölçüde doğru ve eksiksiz yanıtlıyor?",
    "criteria": [
        "Cevap yanlış ya da soruyla ilgisiz.",
        "Cevapta doğru bir parça var ama temel noktalar eksik ya da hatalı.",
        "Cevap büyük ölçüde doğru, küçük bir eksik ya da belirsizlik var.",
        "Cevap doğru ve eksiksiz.",
    ],
}
SUBJECT = {
    "type": "choice",
    "instructions": "Bu soru hangi derse ait?",
    "criteria": dict.fromkeys(SUBJECTS),
}
# Each exam question takes a topic, cycled per subject, so no two requests share
# a prompt; the review found 400 requests over eight prompts, which would have
# filled both splits with the same questions.
TOPICS = {
    "Türkçe": (
        "fiilimsiler",
        "cümlenin ögeleri",
        "anlatım bozuklukları",
        "paragrafta ana fikir",
        "söz sanatları",
        "yazım kuralları",
        "noktalama işaretleri",
        "ses olayları",
        "sözcükte anlam",
        "cümle türleri",
        "fiil çatısı",
        "anlatım biçimleri",
        "düşünceyi geliştirme yolları",
        "ek fiil",
        "zamirler",
    ),
    "Matematik": (
        "olasılık",
        "türev",
        "integral",
        "logaritma",
        "diziler",
        "trigonometri",
        "permütasyon ve kombinasyon",
        "fonksiyonlar",
        "limit",
        "ikinci dereceden denklemler",
        "polinomlar",
        "karmaşık sayılar",
        "analitik geometri",
        "istatistik",
        "mutlak değer",
    ),
    "Fizik": (
        "Newton'un hareket yasaları",
        "iş ve enerji",
        "elektrik akımı",
        "manyetizma",
        "dalgalar",
        "optik",
        "basınç",
        "ısı ve sıcaklık",
        "momentum",
        "atış hareketleri",
        "dairesel hareket",
        "elektrik yükleri",
        "modern fizik",
        "kaldırma kuvveti",
        "basit makineler",
    ),
    "Kimya": (
        "mol kavramı",
        "kimyasal bağlar",
        "asitler ve bazlar",
        "periyodik tablo",
        "gazlar",
        "çözeltiler",
        "tepkime hızı",
        "kimyasal denge",
        "elektrokimya",
        "organik bileşikler",
        "atom modelleri",
        "karışımlar",
        "enerji ve tepkimeler",
        "çözünürlük",
        "hibritleşme",
    ),
    "Biyoloji": (
        "hücre bölünmesi",
        "fotosentez",
        "solunum",
        "kalıtım",
        "ekosistem",
        "sindirim sistemi",
        "dolaşım sistemi",
        "sinir sistemi",
        "hormonlar",
        "evrim",
        "protein sentezi",
        "bitki biyolojisi",
        "enzimler",
        "bağışıklık sistemi",
        "canlıların sınıflandırılması",
    ),
    "Tarih": (
        "Osmanlı'nın kuruluşu",
        "Kurtuluş Savaşı",
        "Tanzimat Fermanı",
        "Malazgirt Savaşı",
        "Lozan Antlaşması",
        "İnkılaplar",
        "Birinci Dünya Savaşı",
        "Coğrafi Keşifler",
        "Rönesans ve Reform",
        "Fransız İhtilali",
        "İstanbul'un fethi",
        "Kanuni dönemi",
        "Meşrutiyet dönemleri",
        "Soğuk Savaş",
        "İlk Türk devletleri",
    ),
    "Coğrafya": (
        "iklim tipleri",
        "nüfus piramitleri",
        "yer şekilleri",
        "levha hareketleri",
        "akarsular",
        "göç",
        "tarım ürünleri",
        "doğal afetler",
        "harita bilgisi",
        "toprak tipleri",
        "enerji kaynakları",
        "şehirleşme",
        "bitki örtüsü",
        "ekonomik faaliyetler",
        "Türkiye'nin konumu",
    ),
    "Felsefe": (
        "bilgi felsefesi",
        "ahlak felsefesi",
        "sanat felsefesi",
        "din felsefesi",
        "siyaset felsefesi",
        "bilim felsefesi",
        "varlık felsefesi",
        "özgürlük",
        "mantığın ilkeleri",
        "Sokrates",
        "Platon",
        "Aristoteles",
        "Descartes",
        "Kant",
        "felsefi düşüncenin özellikleri",
    ),
}
EDU_QUESTIONS = 400

# legal-aid routing --------------------------------------------------------

COURTS = {
    "İş mahkemesi": (
        "işçi ile işveren arasındaki uyuşmazlık",
        (
            "ödenmeyen maaş",
            "kıdem tazminatı",
            "işe iade",
            "fazla mesai ücreti",
            "iş kazası tazminatı",
        ),
    ),
    "Aile mahkemesi": (
        "evlilik, boşanma ve çocuklarla ilgili uyuşmazlık",
        ("boşanma", "nafaka", "velayet", "çocukla kişisel ilişki", "mal paylaşımı"),
    ),
    "Tüketici mahkemesi ya da hakem heyeti": (
        "bir satıcı ya da hizmet sağlayıcıyla tüketici uyuşmazlığı",
        ("ayıplı ürün", "abonelik iptali", "kargo hasarı", "tatil paketi", "garanti reddi"),
    ),
    "Asliye ticaret mahkemesi": (
        "iki tacir ya da şirket arasındaki ticari uyuşmazlık",
        (
            "ödenmeyen fatura",
            "haksız rekabet",
            "ortaklık anlaşmazlığı",
            "tedarik sözleşmesi",
            "ticari satış bedeli",
        ),
    ),
    "Sulh hukuk mahkemesi": (
        "kira, ortaklığın giderilmesi ve benzeri uyuşmazlık",
        (
            "kiracının tahliyesi",
            "kira artışı",
            "miras kalan evin paylaşılması",
            "ödenmeyen kira",
            "kat maliki anlaşmazlığı",
        ),
    ),
    "İdare mahkemesi": (
        "bir kamu kurumunun işlemine itiraz",
        (
            "memur disiplin cezası",
            "imar ruhsatı",
            "atama ve tayin",
            "kamu sınavı sonucu",
            "öğrenci disiplin cezası",
        ),
    ),
    "Asliye hukuk mahkemesi": (
        "başka bir mahkemeye ayrılmamış hukuki uyuşmazlık",
        (
            "tapu iptali",
            "nüfus kaydında ad düzeltme",
            "komşunun tecavüzü",
            "sözleşme ihlali",
            "haksız fiil",
        ),
    ),
    "İcra hukuk mahkemesi": (
        "başlamış bir icra takibi ya da hacizle ilgili şikâyet",
        (
            "maaşa haciz",
            "evdeki eşyaya haciz",
            "haczedilemez eşyanın haczi",
            "başkasının malına haciz",
            "icra memurunun işlemine şikâyet",
        ),
    ),
    "Savcılığa suç duyurusu": (
        "bir suç, ceza soruşturması gerektiren durum",
        ("dolandırılma", "tehdit", "hakaret", "hırsızlık", "darp"),
    ),
}
COURT = {
    "type": "choice",
    "instructions": "Bu kişinin sorunu yargıya taşınırsa hangi mahkeme ya da merci ilgilenir?",
    "criteria": {name: desc for name, (desc, _scenes) in COURTS.items()},
}
LEGAL_PER_COURT = 250

# check-worthiness ---------------------------------------------------------

NON_CLAIM_KINDS = (
    Kind(
        "gorus",
        "bir konuda kişisel görüş ya da değerlendirme; kontrol edilebilecek bir olgu içermez",
        ("siyaset", "futbol", "dizi", "yemek", "şehir hayatı", "eğitim"),
        300,
    ),
    Kind(
        "soru",
        "bir şeyi soran bir paylaşım",
        ("teknoloji", "sağlık", "seyahat", "okul", "alışveriş", "hava durumu"),
        300,
    ),
    Kind(
        "deneyim",
        "kişinin kendi yaşadığını ya da hissettiğini anlattığı paylaşım",
        ("iş günü", "tatil", "hastalık", "trafik", "aile", "spor"),
        300,
    ),
    Kind(
        "dilek",
        "bir dilek, çağrı ya da kutlama",
        ("bayram", "maç", "yeni yıl", "doğum günü", "bağış çağrısı", "geçmiş olsun"),
        300,
    ),
)
CLAIM_KIND = Kind(
    "iddia",
    "kontrol edilebilecek bir olgu iddiası içeren paylaşım (bir sayı, bir olay, birinin "
    "söylediği bir söz); iddia doğru ya da yanlış olabilir",
    ("ekonomi", "sağlık", "spor", "bilim", "yerel haber", "teknoloji"),
    900,
)
HUMAN_QUESTIONS = 900
CHECKABLE = {
    "type": "noul",
    "instructions": (
        "Bu metin, doğruluğu kanıtlarla kontrol edilebilecek bir olgu iddiası içeriyor mu?"
    ),
    "criteria": {
        "true": "Evet, metin kontrol edilebilecek bir olgu iddiası içeriyor.",
        "false": "Hayır, metin görüş, soru, dilek ya da kişisel deneyimden ibaret.",
    },
}
CLAIMS_FROM_FACTURK = 900


def asks() -> list[Ask]:
    messages = {k.name: k for k in MESSAGE_KINDS}
    return [
        Ask("mesaj_turu", "spam", MESSAGE_TYPE,
            {"islem": "işlem bildirimi", "pazarlama": "izinli pazarlama",
             "istenmeyen": "istenmeyen reklam", "dolandirici": "dolandırıcılık",
             "kisisel": "kişisel yazışma"}, tuple(messages)),
        Ask("oltalama", "oltalama", PHISHING,
            {k: ("true" if k == "dolandirici" else "false") for k in messages}, tuple(messages)),
        Ask("cevap_puani", "egitim", GRADE, {f"seviye{i}": str(i) for i in range(4)},
            tuple(f"seviye{i}" for i in range(4))),
        Ask("ders", "egitim", SUBJECT, {f"ders:{s}": s for s in SUBJECTS},
            tuple(f"ders:{s}" for s in SUBJECTS)),
        Ask("basvuru_yeri", "hukuk", COURT, {name: name for name in COURTS}, tuple(COURTS)),
        # Both labels come from both sources, people and the generator, so the
        # writer cannot stand in for the answer: FACTurk claims and
        # generated claims on one side, generated non-claims on the other, and
        # people's own questions from the corpus, left to the judges.
        Ask("dogrulanabilir", "dogrulama", CHECKABLE,
            {"facturk": "true", "iddia": "true", **{k.name: "false" for k in NON_CLAIM_KINDS}},
            ("facturk", "iddia", "mqa_soru", *(k.name for k in NON_CLAIM_KINDS))),
    ]  # fmt: skip


# writing texts --------------------------------------------------------------


def short_id(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()


def scene_split(scenes: tuple[str, ...], scene: str) -> str:
    """Every fifth scene or topic of a kind is validation, whole: texts written to the
    same scene are near-duplicates, and splitting them one by one would put copies
    on both sides."""
    return "validation" if scenes.index(scene) % 5 == 0 else "train"


def split_of(source_id: str) -> str:
    bucket = int(hashlib.blake2b(source_id.encode(), digest_size=4).hexdigest(), 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


NAMES = (
    "Marka, kurum ve kişi adları için uydurma adlar yaz; [marka] gibi yer tutucu kullanma, "
    "yer tutucu olarak yalnızca [bağlantı] ve [telefon] yaz."
)
# Placeholders the masking itself writes; any other bracketed token is a
# writer's placeholder that only one kind would carry, a giveaway.
PLACEHOLDERS = frozenset(("[bağlantı]", "[telefon]", "[ad]", "[e-posta]", "@[kullanıcı]",
                          "[kullanıcı]", "[iban]", "[kimlik]"))  # fmt: skip
BRACKETED = re.compile(r"@?\[[^\]]{1,30}\]")


# Real banks, public bodies, operators, shops and carriers: a generated fraud
# message naming one would be an impersonation template, so such texts are
# dropped from the message and claim kinds.
REAL_NAMES = re.compile(
    r"ziraat|halkbank|vak[ıi]f ?bank|i[şs] ?bankas|garanti|akbank|yap[ıi] ?kredi|qnb|"
    r"denizbank|\bteb\b|\bing\b|enpara|papara|kuveyt t[üu]rk|albaraka|e-?devlet|\bptt\b|"
    r"\bsgk\b|gelir idaresi|\bgib\b|turkcell|vodafone|t[üu]rk telekom|trendyol|hepsiburada|"
    r"amazon|getir|yemeksepeti|aras kargo|yurti[çc]i kargo|\bmng\b|s[üu]rat kargo|"
    r"t[üu]rk hava yollar|\bthy\b|pegasus|\bbim\b|a101|migros|\b[şs]ok market|n11|"
    r"sahibinden|netflix|spotify|apple|google|microsoft|whatsapp|instagram|facebook|"
    r"e-?nab[ıi]z|\bmhrs\b|emniyet|jandarma|i[şs]kur|[öo]sym|\bmeb\b|t[üu]ik|"
    r"sa[ğg]l[ıi]k bakanl|i[çc]i[şs]leri",
    re.IGNORECASE,
)


def clean_placeholders(text: str) -> bool:
    return all(m.group(0) in PLACEHOLDERS for m in BRACKETED.finditer(text))


def message_request(kind: Kind, scene: str, n: int) -> str:
    channel = "SMS" if random.Random(scene + kind.name).random() < 0.5 else "e-posta"
    return (
        f"{n} farklı Türkçe {channel} mesajı yaz. Tür: {kind.brief}. Konu: {scene}. "
        "Her biri gerçekçi, birbirinden farklı üslupta ve uzunlukta olsun; bazıları kısa, "
        "bazıları birkaç cümle. Türün adını mesajın içinde söyleme. " + NAMES + " "
        'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'
    )


def claim_request(kind: Kind, scene: str, n: int) -> str:
    return (
        f"{n} farklı Türkçe sosyal medya paylaşımı yaz. Tür: {kind.brief}. Konu: {scene}. "
        "Her biri kontrol edilebilecek somut bir iddia içersin. " + NAMES + " "
        'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'
    )


def nonclaim_request(kind: Kind, scene: str, n: int) -> str:
    return (
        f"{n} farklı Türkçe sosyal medya paylaşımı yaz. Tür: {kind.brief}. Konu: {scene}. "
        "Doğruluğu kontrol edilebilecek sayı, tarih ya da olay iddiası içermesin. " + NAMES + " "
        'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'
    )


def legal_request(court: str, desc: str, scene: str, n: int) -> str:
    return (
        f"{n} farklı kişinin kendi hukuki sorununu gündelik Türkçeyle anlattığı kısa bir metin "
        f"yaz. Sorunun türü: {desc}; konu: {scene}. Kişi hukuk bilmiyor: mahkeme adı, kanun "
        "maddesi ya da hukuk terimi kullanmasın, yalnızca başına geleni anlatsın. "
        f"(Bu sorun için gidilecek yer: {court}; bunu metinde söyleme.) "
        'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'
    )


def education_request(subject: str, topic: str, angle: int) -> str:
    angles = ("bir tanım ya da kavram", "bir neden-sonuç ilişkisi", "bir karşılaştırma",
              "bir örnek üzerinden açıklama")  # fmt: skip
    return (
        f"lise düzeyinde {subject} dersinden, '{topic}' konusunda, {angles[angle % 4]} "
        "soran açık uçlu bir sınav sorusu yaz ve bu soruya "
        "dört öğrencinin cevabını yaz: seviye 0 yanlış ya da ilgisiz, seviye 1 kısmen doğru ama "
        "temel noktalar eksik, seviye 2 büyük ölçüde doğru ama küçük bir eksik var, seviye 3 "
        "doğru ve eksiksiz. Cevaplar öğrenci diliyle ve farklı uzunlukta olsun; seviyeyi "
        'cevabın içinde söyleme. Yalnızca şu JSON nesnesini yaz: {"soru": "...", '
        '"cevaplar": [{"seviye": 0, "metin": "..."}, ...]}'
    )


def parse_json(content: str):
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def generation_plan(seed: int, scale: float = 1.0) -> list[dict]:
    """Every generator request, each with the kind of text it should return.

    `scale` keeps that share of each kind's requests, for a smoke run.
    """
    rng = random.Random(seed)
    plan = []
    for kind in MESSAGE_KINDS:
        for i in range(kind.count // BATCH):
            scene = kind.scenes[i % len(kind.scenes)]
            split = scene_split(kind.scenes, scene)
            # One writer for every message kind, so style cannot mark the fraud
            # ones; judge A's model, since the generator's provider refuses to
            # write fraud messages at all.
            plan.append({"id": f"msg-{kind.name}-{i}", "kind": kind.name, "shape": "list",
                         "writer": "A", "split": split, "brand_check": True,
                         "prompt": message_request(kind, scene, BATCH)})  # fmt: skip
    for kind in NON_CLAIM_KINDS:
        for i in range(kind.count // BATCH):
            scene = kind.scenes[i % len(kind.scenes)]
            plan.append({"id": f"nc-{kind.name}-{i}", "kind": kind.name, "shape": "list",
                         "split": scene_split(kind.scenes, scene),
                         "prompt": nonclaim_request(kind, scene, BATCH)})  # fmt: skip
    for i in range(CLAIM_KIND.count // BATCH):
        scene = CLAIM_KIND.scenes[i % len(CLAIM_KIND.scenes)]
        plan.append({"id": f"claim-{i}", "kind": CLAIM_KIND.name, "shape": "list",
                     "split": scene_split(CLAIM_KIND.scenes, scene), "brand_check": True,
                     "prompt": claim_request(CLAIM_KIND, scene, BATCH)})  # fmt: skip
    for court, (desc, scenes) in COURTS.items():
        for i in range(LEGAL_PER_COURT // BATCH):
            plan.append(
                {
                    "id": f"law-{short_id(court)}-{i}",
                    "kind": court,
                    "shape": "list",
                    "split": scene_split(scenes, scenes[i % len(scenes)]),
                    "prompt": legal_request(court, desc, scenes[i % len(scenes)], BATCH),
                }
            )
    for i in range(EDU_QUESTIONS):
        subject = SUBJECTS[i % len(SUBJECTS)]
        topics = TOPICS[subject]
        topic = topics[(i // len(SUBJECTS)) % len(topics)]
        plan.append({"id": f"edu-{i}", "kind": f"ders:{subject}", "shape": "exam",
                     "split": scene_split(topics, topic),
                     "prompt": education_request(subject, topic, i // (len(SUBJECTS) * len(topics)))})  # noqa: E501  # fmt: skip
    if scale < 1.0:
        by_kind = defaultdict(list)
        for request in plan:
            by_kind[request["kind"]].append(request)
        plan = [r for rs in by_kind.values() for r in rs[: max(1, round(len(rs) * scale))]]
    rng.shuffle(plan)
    return plan


def texts_from(request: dict, content: str) -> list[dict]:
    """The texts one generator answer holds, each with its intended kind; [] if unusable."""
    try:
        data = parse_json(content)
    except (json.JSONDecodeError, IndexError):
        return []
    out = []
    if request["shape"] == "list":
        for item in data if isinstance(data, list) else []:
            text = str(item.get("metin", "")).strip() if isinstance(item, dict) else ""
            masked = mask_personal_data(text)
            if not (15 <= len(text) <= 1500 and clean_placeholders(masked)):
                continue
            if request.get("brand_check") and REAL_NAMES.search(masked):
                continue
            out.append({"kind": request["kind"], "text": masked, "split": request.get("split"),
                        "request": request["id"]})  # fmt: skip
        return out
    if not isinstance(data, dict):
        return []
    question = str(data.get("soru", "")).strip()
    answers = data.get("cevaplar") or []
    try:
        levels = {int(a.get("seviye", -1)): str(a.get("metin", "")).strip()
                  for a in answers if isinstance(a, dict)}  # fmt: skip
    except (TypeError, ValueError):
        return []
    if not question or sorted(levels) != [0, 1, 2, 3] or not all(levels.values()):
        return []
    exam = short_id(question)
    for level, answer in levels.items():
        out.append(
            {
                "kind": f"seviye{level}",
                "exam": exam,
                "split": request.get("split"),
                "request": request["id"],
                "subject": request["kind"],
                "text": mask_personal_data(f"Soru: {question}\nÖğrenci cevabı: {answer}"),
            }
        )
    return out


def write_texts(
    client, writers: dict, out: Path, seed: int, workers: int, scale: float = 1.0
) -> list[dict]:
    """Generated texts plus FACTurk claims, journaled per request so a restart resumes."""
    path = out / "generated.jsonl"
    done = {r["request"]: r for r in read_jsonl(path)}
    plan = [r for r in generation_plan(seed, scale) if r["id"] not in done]

    def ask(request):
        writer = writers[request.get("writer", "G")]
        try:
            response = client.chat(
                writer["model"], writer["provider"],
                [{"role": "system", "content": GENERATOR_SYSTEM},
                 {"role": "user", "content": request["prompt"]}],
                max_tokens=3000, temperature=0.9, forbid_reasoning_tokens=False,
                reasoning_effort=writer.get("reasoning", "none"),
            )  # fmt: skip
        except STOPPING:
            raise
        except Exception as error:
            if is_transient(error):
                return None
            return {"request": request["id"], "texts": [], "error": type(error).__name__}
        try:
            content = response["choices"][0]["message"]["content"] or ""
            return {"request": request["id"], "texts": texts_from(request, content)}
        except Exception as error:  # a malformed answer is journaled, never fatal
            return {"request": request["id"], "texts": [], "error": type(error).__name__}

    stopped = None
    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(ask, request) for request in plan]
        for future in as_completed(futures):
            try:
                result = future.result()
            except STOPPING as error:
                stopped = error
                continue
            if result is not None:
                done[result["request"]] = result
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    if stopped is not None:
        raise stopped
    texts, seen = [], set()
    for result in done.values():
        for t in result["texts"]:
            key = " ".join(t["text"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            sid = short_id(t["text"])
            # The split set with the request (by scene or topic); a text journaled
            # without one falls back to its hash.
            split = t.get("split") or split_of(t.get("exam") or sid)
            texts.append({**t, "source_id": sid, "source": SOURCE, "split": split})
    return (texts + facturk_claims(seed, round(CLAIMS_FROM_FACTURK * scale))
            + human_questions(seed, round(HUMAN_QUESTIONS * scale)))  # fmt: skip


def human_questions(seed: int, count: int = HUMAN_QUESTIONS) -> list[dict]:
    """People's own questions from the corpus sample, in the build's split of their text."""
    from data.label.build import text_pool
    from data.label.texts import still_allowed

    path = Path("results/step3/build/texts.jsonl")
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    out = []
    for row in rows:
        if not still_allowed(row):
            continue
        question = row["text"].split("\nCevap: ", 1)[0].removeprefix("Soru: ").strip()
        if 15 <= len(question) <= 500:
            split = "validation" if text_pool(row["source_id"]) == "eval" else "train"
            out.append({"kind": "mqa_soru", "text": question, "split": split,
                        "source_id": f"q-{row['source_id']}",
                        "source": f"clips/mqa:{row['config']}"})  # fmt: skip
    random.Random(seed).shuffle(out)
    return out[:count]


def facturk_claims(seed: int, count: int = CLAIMS_FROM_FACTURK) -> list[dict]:
    """Checkable claims from the FACTurk rows already built, in their own split."""
    rows = []
    for split in ("train", "validation"):
        path = Path(f"data/built/typed/facturk_verdict/{split}.jsonl")
        for line in path.read_text("utf-8").splitlines() if path.exists() else []:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "kind": "facturk",
                        "text": row["state"],
                        "split": split,
                        "source_id": row["row_id"],
                        "source": row["source"],
                    }
                )
    random.Random(seed).shuffle(rows)
    return rows[:count]


# labelling ----------------------------------------------------------------


def label_one(client, judges, ask_: Ask, text: dict) -> dict | None:
    question = question_of(ask_.question)
    keys = outcomes(question)
    draft = TrainingRow(
        track=ask_.track, task=f"{ask_.track}-{ask_.task}", split=text["split"],
        origin="generated", label_kind="rule", source=text["source"], state=text["text"],
        question=question, target={k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)},
        recipe=RECIPE,
    )  # fmt: skip
    record = {"key": f"{ask_.task}:{text['source_id']}", "task": ask_.task,
              "kind": text["kind"], "expected": ask_.expected.get(text["kind"])}  # fmt: skip
    try:
        for attempt in range(2):
            try:
                seed = draft.row_id if attempt == 0 else f"{draft.row_id}-again"
                votes = label_row(client, judges, seed, text["text"], question)
                break
            except LowLabelMass as error:
                if attempt == 1:
                    return record | {"outcome": "low_mass", "judge": error.judge}
        row = TrainingRow(**{**draft.model_dump(mode="json"), "label_kind": "judges",
                             "judges": [v.model_dump() for v in votes],
                             "target": mean_vote(votes, keys), "row_id": ""})  # fmt: skip
        outcome = "no_fit" if no_fit(votes) else "labelled"
        return record | {"outcome": outcome, "row": row.model_dump(mode="json")}
    except STOPPING:
        raise
    except Exception as error:
        if is_transient(error):
            return None
        return record | {"outcome": f"failed:{type(error).__name__}"}


def pairs(texts: list[dict]) -> list[tuple[Ask, dict]]:
    out = []
    by_exam = defaultdict(list)
    for t in texts:
        if "exam" in t:
            by_exam[t["exam"]].append(t)
    for ask_ in asks():
        for t in texts:
            if ask_.task == "ders":
                # The subject is asked once per exam question, on its full answer.
                if t.get("kind") == "seviye3":
                    out.append((ask_, {**t, "kind": t["subject"]}))
                continue
            if t["kind"] in ask_.only:
                out.append((ask_, t))
    return out


def run(
    client,
    judges,
    generator,
    out: Path,
    seed: int,
    workers: int,
    scale: float = 1.0,
    relabel: bool = False,
) -> dict:
    a = next(j for j in judges if j.name == "A")
    writers = {"G": generator, "A": {"model": a.model, "provider": a.provider,
                                     "reasoning": a.reasoning}}  # fmt: skip
    texts = write_texts(client, writers, out, seed, workers, scale)
    journal_path = out / "journal.jsonl"
    records = {r["key"]: r for r in read_jsonl(journal_path)}
    todo = [(a, t) for a, t in pairs(texts) if f"{a.task}:{t['source_id']}" not in records]
    stopped = None
    with ThreadPoolExecutor(workers) as pool, journal_path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(label_one, client, judges, a, t) for a, t in todo]
        for future in futures:
            try:
                record = future.result()
            except STOPPING as error:
                stopped = error
                continue
            if record is not None:
                records[record["key"]] = record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    if stopped is not None:
        raise stopped
    if relabel:
        judge_b = load_relabel_judge()
        records = relabel_without_writer(client, judge_b, records, journal_path, workers)
    return assemble(out, texts, records, client)


# Judge A wrote every message, and on the fraud question its confidence on its
# own texts ran 0.15 above judge C's, against 0.03 or less on texts another
# model wrote. The message tasks are relabelled by B, the other family's
# latest release, beside C's existing vote, so no writer grades its own text.
# B's route lives in config/panel.json with the rest of the panel.
WRITER_TASKS = ("mesaj_turu", "oltalama")


def relabel_without_writer(client, judge_b, records: dict, journal_path: Path,
                           workers: int) -> dict:  # fmt: skip
    """Replace judge A's vote on the writer's own texts with judge B's."""

    def one(record):
        row = TrainingRow.model_validate(record["row"])
        c_vote = next(v for v in row.judges if v.judge == "C")
        try:
            b_vote = label_row(client, [judge_b], f"{row.row_id}-b", text_of(row.state),
                               row.question)[0]  # fmt: skip
        except STOPPING:
            raise
        except LowLabelMass:
            return record | {"outcome": "low_mass", "judge": "B", "relabelled": True}
        except Exception as error:
            if is_transient(error):
                return None
            return record | {"outcome": f"failed:{type(error).__name__}", "relabelled": True}
        keys = outcomes(row.question)
        votes = [b_vote, c_vote]
        new = TrainingRow(**{**row.model_dump(mode="json"), "row_id": "",
                             "judges": [v.model_dump() for v in votes],
                             "target": mean_vote(votes, keys)})  # fmt: skip
        outcome = "no_fit" if no_fit(votes) else "labelled"
        return record | {"outcome": outcome, "row": new.model_dump(mode="json"),
                         "relabelled": True}  # fmt: skip

    todo = [r for r in records.values() if r["task"] in WRITER_TASKS and r.get("row")
            and not r.get("relabelled")]  # fmt: skip
    stopped = None
    with ThreadPoolExecutor(workers) as pool, journal_path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(one, r) for r in todo]
        for future in as_completed(futures):
            try:
                record = future.result()
            except STOPPING as error:
                stopped = error
                continue
            if record is not None:
                records[record["key"]] = record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    if stopped is not None:
        raise stopped
    return records


def assemble(out: Path, texts: list[dict], records: dict, client) -> dict:
    by_task = defaultdict(list)
    for r in records.values():
        by_task[r["task"]].append(r)
    rows_out, surplus, nofit, report = [], [], [], {}
    for ask_ in asks():
        recs = by_task.get(ask_.task, [])
        rows = [TrainingRow.model_validate(r["row"]) for r in recs if r["outcome"] == "labelled"]
        nofit += [TrainingRow.model_validate(r["row"]) for r in recs if r["outcome"] == "no_fit"]
        kept, over = balance([r for r in rows if r.split == "train"],
                             CAP_TOP_SHARE[ask_.question["type"]])  # fmt: skip
        rows_out += kept + [r for r in rows if r.split != "train"]
        surplus += over
        agree = [top_of(TrainingRow.model_validate(r["row"]).target) == r["expected"]
                 for r in recs if r["outcome"] == "labelled" and r.get("expected")]  # fmt: skip
        # Judge A wrote every message, so it grades its own texts there; the two
        # judges' agreement and confidence, beside the other tasks', show
        # whether that inflates its votes.
        labelled = [TrainingRow.model_validate(r["row"]) for r in recs
                    if r["outcome"] == "labelled"]  # fmt: skip
        tops = [{v.judge: top_of(v.distribution) for v in row.judges} for row in labelled]
        conf = defaultdict(list)
        for row in labelled:
            for v in row.judges:
                conf[v.judge].append(max(v.distribution.values()))
        report[ask_.task] = {
            "judges_agree_on_top": (
                round(sum(len(set(t.values())) == 1 for t in tops) / len(tops), 4) if tops else None
            ),
            "mean_top_probability": {j: round(sum(c) / len(c), 4) for j, c in conf.items()},
            "pairs": len(recs),
            "outcomes": dict(Counter(r["outcome"] for r in recs)),
            "rows_by_split": dict(
                Counter(r.split for r in kept + [r for r in rows if r.split != "train"])
            ),
            "surplus": len(over),
            "judges_agree_with_intended_kind": round(sum(agree) / len(agree), 4) if agree else None,
            "top_answers_train": dict(Counter(top_of(r.target) for r in kept)),
        }
    for name, rows in (("rows", rows_out), ("surplus", surplus), ("nofit", nofit)):
        (out / f"{name}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows),
                                           "utf-8")  # fmt: skip
    return {
        "texts": len(texts),
        "texts_by_kind": dict(Counter(t["kind"] for t in texts)),
        "tasks": report,
        "rows": len(rows_out),
        "spent_usd": round(client.spent, 4),
        "usd_per_row": round(client.spent / len(rows_out), 5) if rows_out else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.synth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--relabel-messages",
        action="store_true",
        help="replace the writer's own votes on the message tasks",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="share of every kind, for a smoke run"
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = generation_plan(args.seed, args.scale)
        print(
            json.dumps(
                {
                    "requests": len(plan),
                    "by_kind": dict(Counter(p["kind"] for p in plan)),
                    "facturk_claims": len(facturk_claims(args.seed)),
                    "asks": [a.task for a in asks()],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    judges, generator, _cheap = load_panel()
    client = make_client(args.out, args.cap_usd)
    try:
        result = run(
            client,
            judges,
            generator,
            args.out,
            args.seed,
            args.workers,
            args.scale,
            relabel=args.relabel_messages,
        )
    except STOPPING as error:
        print(f"stopped: {error}; restart on the same --out to resume", file=sys.stderr)
        return 1
    (args.out / "build.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
