"""HakemBench test items for the generated tracks and the support and moderation halves.

    python -m bench.hakembench.generate --dry-run          # the plan and its cost, no calls
    python -m bench.hakembench.generate --sample-private   # the corpus halves, no paid calls
    python -m bench.hakembench.generate --cap-usd 12       # write, check and label

Runs where the API key lives, like data/label/synth.py, whose machinery it
imports: the generator requests, the parsing and masking of what comes back,
the questions, and the panel's labelling of each pair (synth.label_one).

What is new here is the texts. Every text is written for this benchmark on
topics and scenes the training texts never had, and each list is checked
against synth's at import:

- egitim: five new topics per subject, one exam question with four answers at
  the four quality levels; asked the grade and the subject.
- spam: synth's five message kinds, eight new scenes each; asked the message
  type and the fraud question.
- hukuk: 120 new court-routing scenes, one lay description each; asked the
  court question.
- moderasyon: forum replies, product reviews and comments, a planned mix of
  offensive and not; asked OffensEval-TR's question word for word.
- sss: customer-support question-and-answer pairs over eight sectors; asked
  one question per held-out cell of the build, cells no training row ever had.

The private halves of moderasyon and sss are drawn from clips/mqa through
data/label/texts.py (its sampling and every filter), leaving out any text that
is already in data/built, in the build's text sample or in HakemBench-dev.
Q&A-site text is disclaimed by its pages, so it stays in the private half.

Before any pair is labelled, generated texts go through texts.py's filters
(adult sites, gambling, stubs, pointer answers) and every text through the
overlap rule of bench/dev.py training_overlap against every file under
data/built. What either removes is counted in the summary.

Judges: a text is never judged by the model that wrote it. Messages and
moderation texts are written by judge A's model, as synth's messages were, so
judge B stands in for A there and the message tasks are always judged by B and
C. The rest are written by the generator and judged by A and C.

Output, per track: candidates/<track>.jsonl in the harness's Item format with
gold empty (the owner sets it on the desk), and <track>.meta.jsonl with the
source, licence, public or private half, the intended kind, the text's sha256
and the panel's votes per question. egitim, spam and hukuk are split in half by
hash within each kind (egitim by the exam question, so its four answers stay
together); generated moderasyon and sss texts are public, corpus texts private.

Everything is journaled; a restart on the same --work pays for nothing twice,
and the client's cap counts the whole step 8 ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from bench.dev import text_hash, training_overlap
from bench.harness.items import Item
from data.decontam.ngrams import NgramIndex, gram_hashes, tokens
from data.label import synth
from data.label import texts as T
from data.label.build import STOPPING, held_out_cells, is_transient, read_jsonl
from data.label.generate import Template, to_question
from data.label.judge import JudgeSpec
from data.label.pilot import load_panel, load_relabel_judge, make_client, question_of
from data.typed.instrument import _question
from schema.rows import outcomes, text_of

WORK = Path("results/private/step8/hakembench")
# One ledger for every paid run of step 8, so the cap covers the step.
LEDGER_DIR = Path("results/private/step8")
CANDIDATES = Path("bench/hakembench/candidates")
DATA = Path("data/built")
TEMPLATES = Path("results/step3/build/templates.jsonl")
BUILD_REPORT = Path("results/step3/build/build.json")
BUILD_TEXTS = Path("results/step3/build/texts.jsonl")
DEV_MANIFEST = Path("results/step6/hakembench_dev_v0.json")
CAP_USD = 12.0

SOURCE = "generated:hakembench-v1"
LICENCE = {
    "generated": "CC BY 4.0 where rights exist, otherwise CC0; "
    "authored by ufak AI for the benchmark",
    "clips/mqa": "private only: clips/mqa (CC0-1.0), page text disclaimed",
}
TRACKS = ("egitim", "spam", "hukuk", "moderasyon", "sss")
# Tracks whose items are all generated and split in half by hash.
HASH_SPLIT = ("egitim", "spam", "hukuk")
PRIVATE_TRACKS = ("moderasyon", "sss")
PRIVATE_COUNT = 160
PRIVATE_SEED = 8
# A labelled pair's split field; the rows are never training rows.
LABEL_SPLIT = "heldout_task"

# Per-call cost by role, the means of the synth run's ledger, from the writer's, the judges' and the
# generator's own calls.
COST_PER_CALL = {"G": 0.00242, "writer A": 0.00104, "A": 0.000102, "B": 0.000751,
                 "C": 0.000193}  # fmt: skip
# Second attempts on a low letter mass and malformed answers retried by hand.
RETRY_MARGIN = 1.25

# egitim -------------------------------------------------------------------

EDU_TOPICS = {
    "Türkçe": ("isim tamlamaları", "fiil kipleri", "edebi türler", "yapım ve çekim ekleri",
               "bağlaçlar"),
    "Matematik": ("üslü sayılar", "köklü sayılar", "kümeler", "eşitsizlikler", "vektörler"),
    "Fizik": ("sürtünme kuvveti", "tork ve denge", "kütle çekimi",
              "elektromanyetik indüksiyon", "özkütle"),
    "Kimya": ("radyoaktivite", "polimerler", "maddenin halleri", "soy gazlar",
              "yükseltgenme ve indirgenme"),
    "Biyoloji": ("boşaltım sistemi", "destek ve hareket sistemi", "duyu organları",
                 "biyoteknoloji", "virüsler ve bakteriler"),
    "Tarih": ("Selçuklu Devleti", "Haçlı Seferleri", "Lale Devri", "Sanayi Devrimi",
              "Mondros Ateşkes Antlaşması"),
    "Coğrafya": ("göller", "rüzgârlar", "madenler", "ulaşım yolları", "kıyı tipleri"),
    "Felsefe": ("Farabi", "Spinoza", "John Locke", "Nietzsche", "adalet kavramı"),
}  # fmt: skip
TOPICS_PER_SUBJECT = 5

# spam ---------------------------------------------------------------------

MESSAGE_BATCH = 8
MESSAGE_SCENES = {
    "islem": ("iade tutarı hesaba geçti", "kütüphane kitap iade tarihi",
              "araç muayene randevusu", "eczanede reçete hazır", "planlı su kesintisi",
              "otopark ödeme makbuzu", "kurs kayıt onayı", "teknik serviste cihaz hazır"),
    "pazarlama": ("kahve zinciri yeni menü", "spor salonu üyelik indirimi",
                  "sinema bileti fırsatı", "elektronik mağazası sezon indirimi",
                  "eczane zinciri vitamin kampanyası", "akaryakıt istasyonu puan kampanyası",
                  "çevrim içi kurs indirimi", "mobilya mağazası açılış duyurusu"),
    "istenmeyen": ("saç ekimi kliniği", "güneş paneli teklifi", "sigorta teklifi",
                   "diyet çayı", "ikinci el araç alımı", "halı yıkama", "yurt dışı dil okulu",
                   "mobil oyun reklamı"),
    "dolandirici": ("sahte trafik cezası ödemesi", "elektrik kesilecek uyarısı",
                    "kapora isteyen sahte kiralık ev", "sahte kripto yatırım fırsatı",
                    "yakınınız kaza geçirdi diye para isteme", "sahte sosyal yardım başvurusu",
                    "sahte ikinci el satıcı", "kargo adresini güncelleme bağlantısı"),
    "kisisel": ("tatil planı", "ev taşıma yardımı", "evcil hayvan bakımı", "düğün hazırlığı",
                "hasta ziyareti", "maç izleme", "ödünç kitap", "yemek tarifi"),
}  # fmt: skip
MESSAGE_KINDS = tuple(
    synth.Kind(k.name, k.brief, MESSAGE_SCENES[k.name], MESSAGE_BATCH * 8)
    for k in synth.MESSAGE_KINDS
)

# hukuk --------------------------------------------------------------------

# One lay description per scene. Each forum was read against the court's
# statutory remit when the scene was written; the owner's gold is what counts.
LEGAL_PER_SCENE = 1
COURT_SCENES = {
    "İş mahkemesi": (
        "ödenmeyen yıllık izin ücreti",
        "ihbar tazminatı",
        "işyerinde mobbing",
        "hafta tatili çalışmasının ücreti",
        "bordroda gösterilmeyen prim",
        "sigortasız geçen çalışmanın tespiti",
        "bayram günü çalışmasının ücreti",
        "ödenmeyen yol ve yemek parası",
        "hamilelik nedeniyle işten çıkarılma",
        "gece çalışmasının ücreti",
        "meslek hastalığı tazminatı",
        "asgari ücretin altında ödenen maaş",
        "ödenmeyen satış komisyonu",
        "ödenmeyen ikramiye",
    ),
    "Aile mahkemesi": (
        "düğünde takılan ziynet eşyalarının geri istenmesi",
        "soybağının reddi",
        "babalık davası",
        "evlat edinme",
        "evliliğin iptali",
        "aile içi şiddete karşı koruma kararı",
        "eşlerin ayrı yaşamasına karar verilmesi",
        "yoksulluk nafakasının kaldırılması",
        "aile konutu şerhi",
        "yurt dışında alınan boşanma kararının tanınması",
        "küçüğün evlenmesine izin",
        "evlat edinmenin kaldırılması",
        "nişan bozulunca hediyelerin geri istenmesi",
    ),
    "Tüketici mahkemesi ya da hakem heyeti": (
        "galeriden alınan arızalı ikinci el araba",
        "internetten alınan ürünü iade etme hakkı",
        "kredi dosya masrafının iadesi",
        "kredi kartı yıllık ücretinin iadesi",
        "özel okul ücretinin geri istenmesi",
        "spor salonu üyeliğinin bitirilmesi",
        "müteahhidin konutu geç teslim etmesi",
        "kuru temizlemede zarar gören giysi",
        "düğün salonu sözleşmesi",
        "iptal edilen uçuşun bilet parası",
        "telefon tamirinde çıkan yeni arıza",
        "özel hastanenin fazla aldığı ücret",
        "internetten verilen siparişin hiç gelmemesi",
    ),
    "Asliye ticaret mahkemesi": (
        "anonim şirket genel kurul kararının iptali",
        "limited şirket ortağının şirketten çıkması",
        "franchise sözleşmesinin bozulması",
        "acentelik sözleşmesi bitince denkleştirme tazminatı",
        "nakliye firmasının taşıdığı yükün hasar görmesi",
        "şirketin finansal kiralama sözleşmesi",
        "şirket yöneticisinin şirketi zarara uğratması",
        "ticari işletmenin devri",
        "iki firma arasındaki cari hesap alacağı",
        "yazılım firmasına ödenmeyen proje bedeli",
        "bayilik sözleşmesinin haksız feshi",
        "borcunu ödeyemeyen şirketin iflası",
        "şirketin konkordato istemesi",
        "şirket deposundaki yangın için sigorta ödemesi",
    ),
    "Sulh hukuk mahkemesi": (
        "kiracının depozitosunu geri alamaması",
        "kiralık evde yapılan tadilatın masrafı",
        "ortak tarlanın satılarak paylaşılması",
        "yaşlı babaya vasi atanması",
        "mirasın reddi",
        "mirasçılık belgesi alınması",
        "apartman yöneticisinin aidat hesabı vermemesi",
        "üst kattaki komşunun balkonundan su akması",
        "vasiyetnamenin açılması",
        "terekenin tespiti",
        "rutubetli kiralık ev için kira indirimi",
        "kiracının aidat ve fatura borcu bırakıp çıkması",
        "kayıp kişinin malları için kayyım atanması",
    ),
    "İdare mahkemesi": (
        "belediyenin işyeri ruhsatını iptal etmesi",
        "üniversiteden kaydın silinmesi",
        "belediyenin yıkım kararı",
        "memur maaşından yapılan kesinti",
        "silah ruhsatı başvurusunun reddi",
        "pasaport verilmemesi",
        "vatandaşlık başvurusunun reddi",
        "kamu ihalesinden elenme",
        "devlet bursunun kesilmesi",
        "imar planının değiştirilmesi",
        "çevresel etki değerlendirmesi olumlu kararı",
        "yabancının sınır dışı edilmesi kararı",
        "devlet hastanesinde yapılan tedavi hatası",
        "devlet okuluna kayıt yapılmaması",
    ),
    "Asliye hukuk mahkemesi": (
        "trafik kazasında karşı sürücüden tazminat",
        "internette hakaret için manevi tazminat",
        "arkadaşa verilen borcun geri alınması",
        "nüfus kaydında yaş düzeltme",
        "kayıp kişi için gaiplik kararı",
        "komşu inşaatının evde çatlak açması",
        "iki kişi arasında satılan arabanın gizli arızası",
        "vasiyetnamenin iptali",
        "miras payı için tenkis",
        "izinsiz kullanılan arsa için ecrimisil",
        "komşunun köpeğinin ısırması için tazminat",
        "nüfus kaydında doğum yerinin düzeltilmesi",
        "satış vaadi sözleşmesine dayanarak tapu tescili",
    ),
    "İcra hukuk mahkemesi": (
        "evin icradan satışında ihalenin feshi",
        "icra takibindeki borcun zamanaşımına uğraması",
        "oturulan tek evin haczi",
        "ödeme emrine yapılan itirazın kaldırılması",
        "sıra cetveline itiraz",
        "ödeme emrinin usulsüz tebliği",
        "senetteki imzanın borçluya ait olmaması",
        "haczedilen malın değerinin düşük biçilmesi",
        "icra dosyasındaki faizin yanlış hesaplanması",
        "nafaka yatan banka hesabına haciz",
        "ödenmiş borç için sürdürülen icra takibi",
        "ipotekli evin paraya çevrilmesi takibi",
        "icra dosyasında fazla alınan paranın iadesi",
    ),
    "Savcılığa suç duyurusu": (
        "adına sahte belgeyle kredi çekilmesi",
        "ısrarlı takip",
        "evine izinsiz girilmesi",
        "arabasının kasten çizilmesi",
        "izinsiz fotoğrafının paylaşılması",
        "şantaj",
        "kişisel verilerinin ele geçirilmesi",
        "internetten cinsel içerikli mesajlarla taciz",
        "sosyal medya hesabının çalınması",
        "zorla telefonunun alınması",
        "hayvana eziyet",
        "memurun rüşvet istemesi",
        "fabrikanın dereye atık dökmesi",
    ),
}
LEGAL_SCENES = 120

# moderasyon ---------------------------------------------------------------

MODERATION_BATCH = 8
MODERATION_FORMS = ("forum yanıtı", "ürün yorumu", "haber sitesine okur yorumu", "video yorumu",
                    "restoran değerlendirmesi")  # fmt: skip
MODERATION_KINDS = (
    synth.Kind("kufur_hakaret", "başka bir yorumcuya, bir satıcıya ya da bir kişiye yönelik küfür "
               "ya da hakaret içeren bir yorum; bir etnik köken, din, cinsiyet ya da benzeri bir "
               "grubu hedef almasın", MODERATION_FORMS, 40),
    synth.Kind("saldirgan", "küfür içermeyen ama karşısındakini aşağılayan, tehdit eden ya da "
               "düşmanca saldıran bir yorum; bir grubu hedef almasın", MODERATION_FORMS, 40),
    synth.Kind("sert_elestiri", "sert ve olumsuz ama küfür, hakaret ya da kişisel saldırı "
               "içermeyen bir eleştiri", MODERATION_FORMS, 40),
    synth.Kind("olagan", "olağan, nazik ya da tarafsız bir yorum; bazıları olumlu, bazıları bir "
               "şey soruyor", MODERATION_FORMS, 40),
)  # fmt: skip
# OffensEval-TR's question, word for word, from its converter.
OFFENSIVE = _question("offenseval_tr", ["NOT", "OFF"])[0]
OFFENSIVE_ASK = synth.Ask(
    "offenseval_tr", "moderasyon", OFFENSIVE,
    {"kufur_hakaret": "true", "saldirgan": "true", "sert_elestiri": "false", "olagan": "false"},
    tuple(k.name for k in MODERATION_KINDS),
)  # fmt: skip

# sss ----------------------------------------------------------------------

SUPPORT_BATCH = 5
SUPPORT_DOMAINS = ("e-ticaret ve kargo", "bankacılık ve ödeme", "telefon ve internet operatörü",
                   "web barındırma ve yazılım", "özel sağlık kliniği", "seyahat ve ulaşım",
                   "sigorta", "özel kurs ve eğitim")  # fmt: skip
SUPPORT_KINDS = (
    synth.Kind("tam", "cevap soruyu doğrudan ve eksiksiz yanıtlıyor", SUPPORT_DOMAINS, 40),
    synth.Kind("kismi", "cevap soruyla ilgili ama önemli bir kısmını yanıtsız bırakıyor",
               SUPPORT_DOMAINS, 40),
    synth.Kind("yonlendiren", "soru kişinin hesabına, ödemesine, sağlığına ya da hukuki bir "
               "işlemine özgü olduğu için cevap kişiyi bir müşteri temsilcisine ya da uzmana "
               "yönlendiriyor", SUPPORT_DOMAINS, 40),
    synth.Kind("ilgisiz", "cevap genel ve kalıp bir metin, sorulan şeyi yanıtlamıyor",
               SUPPORT_DOMAINS, 40),
)  # fmt: skip
# The build's held-out templates kept rows under these statuses (build.assemble).
KEPT_STATUSES = ("complete", "active", "skewed")


def check_disjoint(
    topics: dict = EDU_TOPICS, scenes: dict = MESSAGE_SCENES, courts: dict = COURT_SCENES
) -> None:
    """Raise if a new topic or scene repeats a training one, or a list is the wrong size."""

    def fold(values) -> list[str]:
        return [" ".join(v.casefold().split()) for v in values]

    def fresh(name: str, new: list[str], old: list[str]) -> None:
        if len(set(fold(new))) != len(new):
            raise ValueError(f"{name}: a new entry is listed twice")
        clash = sorted(set(fold(new)) & set(fold(old)))
        if clash:
            raise ValueError(f"{name}: already in the training texts: {clash}")

    if set(topics) != set(synth.SUBJECTS):
        raise ValueError("egitim: the subjects must be synth's")
    if any(len(v) != TOPICS_PER_SUBJECT for v in topics.values()):
        raise ValueError(f"egitim: {TOPICS_PER_SUBJECT} topics per subject")
    fresh("egitim", [t for v in topics.values() for t in v],
          [t for v in synth.TOPICS.values() for t in v])  # fmt: skip
    if set(scenes) != {k.name for k in synth.MESSAGE_KINDS}:
        raise ValueError("spam: one scene list per synth message kind")
    fresh("spam", [s for v in scenes.values() for s in v],
          [s for k in synth.MESSAGE_KINDS for s in k.scenes])  # fmt: skip
    if set(courts) != set(synth.COURTS):
        raise ValueError("hukuk: the courts must be synth's")
    new = [s for v in courts.values() for s in v]
    if len(new) != LEGAL_SCENES:
        raise ValueError(f"hukuk: {LEGAL_SCENES} scenes, got {len(new)}")
    fresh("hukuk", new, [s for _desc, v in synth.COURTS.values() for s in v])


check_disjoint()


# requests -----------------------------------------------------------------

JSON_LIST = 'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'


def comment_request(kind: synth.Kind, form: str, n: int) -> str:
    return (
        f"{n} farklı Türkçe {form} yaz. Tür: {kind.brief}. Her biri gerçekçi, birbirinden farklı "
        "üslupta ve uzunlukta olsun; bazıları kısa, bazıları birkaç cümle. Türün adını metnin "
        "içinde söyleme. " + synth.NAMES + " " + JSON_LIST
    )


def support_request(kind: synth.Kind, domain: str, n: int) -> str:
    return (
        f"{n} farklı Türkçe müşteri hizmetleri soru-cevap çifti yaz. Sektör: {domain}. Bir "
        f"müşteri soruyor, şirketin destek ekibi cevaplıyor. Cevabın türü: {kind.brief}. "
        "Sorular gerçekçi, birbirinden farklı konularda ve farklı uzunlukta olsun. Cevabın "
        "türünü metinde söyleme. " + synth.NAMES + " "
        'Yalnızca şu biçimde bir JSON dizisi yaz: [{"soru": "...", "cevap": "..."}, ...]'
    )


def generation_plan() -> list[dict]:
    """Every generator request: its track, intended kind, writer and how many texts it asks for."""
    plan = []

    def add(track, kind, scene, shape, writer, n, prompt, **extra):
        plan.append({"id": f"hb-{track}-{synth.short_id(kind + '|' + scene)}", "track": track,
                     "kind": kind, "scene": scene, "shape": shape, "writer": writer, "n": n,
                     "prompt": prompt, **extra})  # fmt: skip

    for subject in synth.SUBJECTS:
        for i, topic in enumerate(EDU_TOPICS[subject]):
            add("egitim", f"ders:{subject}", topic, "exam", "G", 4,
                synth.education_request(subject, topic, i))  # fmt: skip
    # One writer for every message kind, as in synth, so style cannot mark fraud.
    for kind in MESSAGE_KINDS:
        for scene in kind.scenes:
            add("spam", kind.name, scene, "list", "A", MESSAGE_BATCH,
                synth.message_request(kind, scene, MESSAGE_BATCH), brand_check=True)  # fmt: skip
    for court, scenes in COURT_SCENES.items():
        desc = synth.COURTS[court][0]
        for scene in scenes:
            add("hukuk", court, scene, "list", "G", LEGAL_PER_SCENE,
                synth.legal_request(court, desc, scene, LEGAL_PER_SCENE))  # fmt: skip
    # The writer of the fraud messages also writes the insults, for the same
    # reason: the generator's provider refused fraud messages outright.
    for kind in MODERATION_KINDS:
        for form in kind.scenes:
            add("moderasyon", kind.name, form, "list", "A", MODERATION_BATCH,
                comment_request(kind, form, MODERATION_BATCH), brand_check=True)  # fmt: skip
    for kind in SUPPORT_KINDS:
        for domain in kind.scenes:
            add("sss", kind.name, domain, "qa", "G", SUPPORT_BATCH,
                support_request(kind, domain, SUPPORT_BATCH), brand_check=True)  # fmt: skip
    return plan


def texts_of(request: dict, content: str) -> list[dict]:
    """synth.texts_from, with question-and-answer pairs turned into one text each first."""
    if request["shape"] != "qa":
        return synth.texts_from(request, content)
    try:
        data = synth.parse_json(content)
    except (json.JSONDecodeError, IndexError):
        return []
    pairs = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            question = str(item.get("soru", "")).strip()
            answer = str(item.get("cevap", "")).strip()
            if question and answer:
                pairs.append({"metin": f"Soru: {question}\nCevap: {answer}"})
    return synth.texts_from({**request, "shape": "list"}, json.dumps(pairs, ensure_ascii=False))


# journaling ---------------------------------------------------------------


def journaled(jobs: list[Callable[[], dict | None]], path: Path, workers: int) -> list[dict]:
    """Run the jobs, appending each finished record; a budget stop is raised after the rest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out, stopped = [], None
    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(job) for job in jobs]
        for future in as_completed(futures):
            try:
                record = future.result()
            except STOPPING as error:
                stopped = error
                continue
            if record is not None:
                out.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    if stopped is not None:
        raise stopped
    return out


def write_one(client, writers: dict, request: dict) -> dict | None:
    writer = writers[request["writer"]]
    try:
        response = client.chat(
            writer["model"], writer["provider"],
            [{"role": "system", "content": synth.GENERATOR_SYSTEM},
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
        return {"request": request["id"], "texts": texts_of(request, content)}
    except Exception as error:  # a malformed answer is journaled, never fatal
        return {"request": request["id"], "texts": [], "error": type(error).__name__}


def write_texts(client, writers: dict, plan: list[dict], path: Path, workers: int) -> list[dict]:
    done = {r["request"]: r for r in read_jsonl(path)}
    todo = [partial(write_one, client, writers, r) for r in plan if r["id"] not in done]
    for record in journaled(todo, path, workers):
        done[record["request"]] = record
    return collect(plan, done)


def collect(plan: list[dict], done: dict[str, dict]) -> list[dict]:
    """The written texts in plan order, each once, with its track and writer."""
    texts, seen = [], set()
    for request in plan:
        for t in done.get(request["id"], {}).get("texts", []):
            key = " ".join(t["text"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            sha = text_hash(t["text"])
            texts.append({**t, "track": request["track"], "writer": request["writer"],
                          "scene": request["scene"], "shape": request["shape"],
                          "source": "generated", "sha": sha, "source_id": sha[:24]})  # fmt: skip
    return texts


# filters, overlap, halves -------------------------------------------------


def drop_reason(text: dict) -> str | None:
    """Why texts.py's rules drop a text, or None when they keep it."""
    body = text["text"]
    if text.get("shape") == "qa" or text["source"] == "clips/mqa":
        kept = T.still_allowed({"text": body, "domain": text.get("domain")})
    else:
        kept = not T.blocked({}, body, "")
    if kept:
        return None
    if T.ADULT.search(body):
        return "adult"
    brands = (w.lower() for w in T.BET_BRAND.findall(body))
    if T.GAMBLING.search(body) or any(not any(s in w for s in T.BET_STEMS) for w in brands):
        return "gambling"
    if T.POINTER.search(body):
        return "pointer"
    question, _, answer = body.partition("\nCevap: ")
    if answer and T.is_stub(question.removeprefix("Soru: "), answer):
        return "stub"
    return "other"


def built_files(root: Path) -> list[Path]:
    """Every file under data/built: training, validation and held-out rows alike."""
    return sorted(root.rglob("*.jsonl"))


def overlapping(texts: list[dict], files: list[Path]) -> dict[str, list[str]]:
    """Texts that bench/dev.py's rule finds in a training file, with the rows that carry them.

    training_overlap names the training rows; each named row is read again to
    find which texts it matched, by the same rule that named it.
    """
    rows = [{"id": t["source_id"], "text": t["text"]} for t in texts]
    found = training_overlap(rows, files)
    if not found:
        return {}
    index = NgramIndex.build((f"dev:{r['id']}", r["text"]) for r in rows)
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[text_hash(r["text"])].append(f"dev:{r['id']}")
    wanted = {(f["file"], f["row_id"]): f["rule"] for f in found}
    hits = defaultdict(list)
    for path in sorted({Path(f["file"]) for f in found}):
        for row in read_jsonl(path):
            rule = wanted.get((str(path), row["row_id"]))
            if rule is None:
                continue
            state = row["state"]
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            if rule == "whole":
                refs = by_hash[text_hash(text)]
            else:
                _fraction, refs = index.overlap(text)
                if rule == "covers":
                    refs = index.covered_references(set(gram_hashes(tokens(text), index.n)), refs)
            for ref in refs:
                hits[ref.split(":", 1)[1]].append(f"{path}:{row['row_id']}")
    return dict(hits)


def split_key(text: dict) -> str:
    """What a half is drawn on: the exam question for egitim, so its answers stay together."""
    return text.get("exam") or text["sha"]


def stratum(text: dict) -> str:
    return text.get("subject") or text["kind"]


def assign_halves(texts: list[dict]) -> None:
    """Public or private: by hash within each kind for the generated tracks, by source for the
    rest. Keys are ranked by their sha256 and alternate, so each kind splits 50/50."""
    groups = defaultdict(set)
    for t in texts:
        if t["track"] in HASH_SPLIT:
            groups[(t["track"], stratum(t))].add(split_key(t))
    side = {}
    for keys in groups.values():
        ranked = sorted(keys, key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
        for i, key in enumerate(ranked):
            side[key] = "public" if i % 2 == 0 else "private"
    for t in texts:
        if t["track"] in HASH_SPLIT:
            t["split"] = side[split_key(t)]
        else:
            t["split"] = "public" if t["source"] == "generated" else "private"


def prepare(generated: list[dict], private: list[dict], files: list[Path]):
    """Filtered, checked against training, and split; with what was dropped and why."""
    drops, kept, seen = Counter(), [], set()
    for t in generated + private:
        reason = drop_reason(t)
        if reason:
            drops[f"{t['track']}:filter:{reason}"] += 1
            continue
        if t["sha"] in seen:
            drops[f"{t['track']}:duplicate"] += 1
            continue
        seen.add(t["sha"])
        kept.append(t)
    hits = overlapping(kept, files)
    for t in kept:
        if t["source_id"] in hits:
            drops[f"{t['track']}:overlap"] += 1
    kept = [t for t in kept if t["source_id"] not in hits]
    assign_halves(kept)
    return kept, drops, hits


# private halves -----------------------------------------------------------


def known_texts(data_root: Path, build_texts: Path = BUILD_TEXTS,
                dev_manifest: Path = DEV_MANIFEST) -> tuple[set[str], set[str]]:  # fmt: skip
    """sha256 of every text already used, and the corpus ids of the build's sample."""
    hashes, ids = set(), set()
    for path in built_files(data_root):
        hashes.update(text_hash(text_of(row["state"])) for row in read_jsonl(path))
    for row in read_jsonl(build_texts):
        hashes.add(text_hash(row["text"]))
        ids.add(str(row["source_id"]))
    if dev_manifest.is_file():
        hashes.update(json.loads(dev_manifest.read_text(encoding="utf-8"))["text_sha256"])
    return hashes, ids


def sample_private(data_root: Path, *, count: int = PRIVATE_COUNT, seed: int = PRIVATE_SEED,
                   oversample: int = 3, sample: Callable = T.sample_faq,
                   build_texts: Path = BUILD_TEXTS,
                   dev_manifest: Path = DEV_MANIFEST) -> list[dict]:  # fmt: skip
    """Corpus texts for the private halves, none already used anywhere.

    Moderation reads community questions only (people's own posts); support
    reads company FAQ pages and community questions half and half, as the build
    did. Downloads the corpus shards on first use; makes no paid call.
    """
    hashes, ids = known_texts(data_root, build_texts, dev_manifest)
    draws = {
        "moderasyon": [(T.CONFIGS[1], count * oversample, seed)],
        "sss": [(T.CONFIGS[0], count * oversample // 2, seed + 1),
                (T.CONFIGS[1], count * oversample // 2, seed + 2)],
    }  # fmt: skip
    out, seen = [], set()
    for track, plan in draws.items():
        rows = []
        for config, n, s in plan:
            per_domain = 3 if config == T.CONFIGS[0] else max(3, n // 10)
            rows += sample(n, config=config, per_domain=per_domain, seed=s)
        taken = 0
        for row in rows:
            sha = text_hash(row["text"])
            if taken >= count or sha in hashes or sha in seen or str(row["source_id"]) in ids:
                continue
            if not T.still_allowed(row):
                continue
            seen.add(sha)
            taken += 1
            out.append({"track": track, "mqa_id": str(row["source_id"]),
                        "domain": row.get("domain"), "config": row["config"],
                        "text": row["text"]})  # fmt: skip
    return out


FILE_OF = {"hukuk": "hukuk-gen"}


def private_texts(path: Path) -> list[dict]:
    """The corpus halves drawn by --sample-private; a run without them is refused."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run --sample-private first and ship it")
    out = []
    for row in read_jsonl(path):
        sha = text_hash(row["text"])
        out.append({**row, "kind": None, "writer": None, "shape": "qa", "source": "clips/mqa",
                    "sha": sha, "source_id": sha[:24]})  # fmt: skip
    return out


# questions and judges -----------------------------------------------------


def held_out_templates(templates: Path = TEMPLATES, report: Path = BUILD_REPORT) -> dict:
    """One build template per held-out cell: of those whose rows the build kept, the one its
    judges agreed on most, then the best check score."""
    entries = read_jsonl(templates)
    status = json.loads(report.read_text(encoding="utf-8"))["per_template"]

    def kept(tid: str) -> bool:
        s = status.get(tid, {}).get("status", "")
        return s in KEPT_STATUSES or s.startswith("mined_")

    chosen = {}
    for cell in sorted(held_out_cells()):
        pool = [e for e in entries if e["held_out"] and e["status"] == "ready" and kept(e["tid"])
                and f"{e['template']['family']}-{e['template']['type']}" == cell]  # fmt: skip
        if not pool:
            raise ValueError(f"no held-out template of {cell} kept its rows")
        chosen[cell] = max(pool, key=lambda e: (status[e["tid"]].get("unanimous") or 0.0,
                                                e["check"], e["tid"]))  # fmt: skip
    return chosen


def track_asks(templates: dict) -> dict[str, list[synth.Ask]]:
    by_task = {a.task: a for a in synth.asks()}
    support = [synth.Ask(cell, "sss", to_question(Template.model_validate(e["template"])))
               for cell, e in sorted(templates.items())]  # fmt: skip
    return {
        "egitim": [by_task["cevap_puani"], by_task["ders"]],
        "spam": [by_task["mesaj_turu"], by_task["oltalama"]],
        "hukuk": [by_task["basvuru_yeri"]],
        "moderasyon": [OFFENSIVE_ASK],
        "sss": support,
    }


def judges_for(ask_: synth.Ask, text: dict, panel: list[JudgeSpec], judge_b: JudgeSpec,
               writer_models: dict[str, str]) -> list[JudgeSpec]:  # fmt: skip
    """The panel, with judge B in place of any judge whose model wrote the text."""
    writer = writer_models.get(text.get("writer"))
    chosen = [judge_b if j.model == writer else j for j in panel]
    if any(j.model == writer for j in chosen):
        raise ValueError(f"no judge left for a text its own model wrote ({text.get('writer')})")
    if ask_.task in synth.WRITER_TASKS and sorted(j.name for j in chosen) != ["B", "C"]:
        raise ValueError(f"{ask_.task} is judged by B and C, never the writer's judge")
    return chosen


def label_text(ask_: synth.Ask, text: dict) -> dict:
    """The text as synth.label_one reads it; the subject is asked with the subject as intent."""
    source = SOURCE if text["source"] == "generated" else f"clips/mqa:{text['config']}"
    kind = text["subject"] if ask_.task == "ders" else text["kind"]
    return {"kind": kind, "text": text["text"], "source_id": text["source_id"],
            "split": LABEL_SPLIT, "source": source}  # fmt: skip


def label_pairs(client, panel, judge_b, writer_models, asks, texts, path, workers) -> dict:
    records = {r["key"]: r for r in read_jsonl(path)}
    todo = []
    for t in texts:
        for ask_ in asks[t["track"]]:
            if f"{ask_.task}:{t['source_id']}" in records:
                continue
            judges = judges_for(ask_, t, panel, judge_b, writer_models)
            todo.append(partial(synth.label_one, client, judges, ask_, label_text(ask_, t)))
    for record in journaled(todo, path, workers):
        records[record["key"]] = record
    return records


# output -------------------------------------------------------------------


def item_id(text: dict) -> str:
    return f"{text['track']}-{text['sha'][:16]}"


def votes_of(record: dict | None) -> dict:
    if record is None:
        return {"outcome": "pending"}
    out = {"outcome": record["outcome"], "expected": record.get("expected")}
    row = record.get("row")
    if row:
        out["judges"] = {v["judge"]: v["distribution"] for v in row["judges"]}
        out["mean"] = row["target"]
    return out


def write_candidates(texts: list[dict], asks: dict, records: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for track in TRACKS:
        chosen = sorted((t for t in texts if t["track"] == track), key=item_id)
        questions = {a.task: a.question for a in asks[track]}
        items, metas, outcome, agree = [], [], Counter(), defaultdict(list)
        for t in chosen:
            item = Item.model_validate({"id": item_id(t), "track": track, "state": t["text"],
                                        "questions": questions})  # fmt: skip
            votes = {}
            for a in asks[track]:
                record = records.get(f"{a.task}:{t['source_id']}")
                votes[a.task] = votes_of(record)
                outcome[votes[a.task]["outcome"]] += 1
                v = votes[a.task]
                if v.get("expected") and "mean" in v:
                    agree[a.task].append(max(v["mean"], key=v["mean"].get) == v["expected"])
            items.append(item.model_dump_json())
            metas.append(json.dumps({
                "id": item.id, "track": track, "source": t["source"],
                "licence": LICENCE[t["source"]], "split": t["split"], "kind": t.get("kind"),
                "subject": t.get("subject"), "scene": t.get("scene"), "writer": t.get("writer"),
                "request": t.get("request"), "mqa_id": t.get("mqa_id"),
                "text_sha256": t["sha"], "votes": votes,
            }, ensure_ascii=False))  # fmt: skip
        # The converted Constitutional Court items already hold hukuk.jsonl, so the
        # generated court-routing items get a file of their own (same track).
        name = FILE_OF.get(track, track)
        (out / f"{name}.jsonl").write_text("".join(x + "\n" for x in items), "utf-8")
        (out / f"{name}.meta.jsonl").write_text("".join(x + "\n" for x in metas), "utf-8")
        report[track] = {
            "items": len(items),
            "pairs": len(items) * len(questions),
            "by_split": dict(Counter(t["split"] for t in chosen)),
            "by_source": dict(Counter(t["source"] for t in chosen)),
            "outcomes": dict(outcome),
            "panel_agrees_with_intended_kind": {
                task: round(sum(v) / len(v), 4) for task, v in agree.items() if v
            },
        }
    return report


# run ----------------------------------------------------------------------


def run(client, panel, generator, judge_b, *, work: Path, out: Path, files: list[Path],
        templates: dict, workers: int, plan: list[dict] | None = None) -> dict:  # fmt: skip
    if not files:
        raise ValueError("no training files to check against; refusing to write test items")
    a = next(j for j in panel if j.name == "A")
    writers = {"G": generator, "A": {"model": a.model, "provider": a.provider,
                                     "reasoning": a.reasoning}}  # fmt: skip
    writer_models = {name: w["model"] for name, w in writers.items()}
    plan = generation_plan() if plan is None else plan
    # The corpus halves are read before anything is paid for, so a run shipped
    # without them stops here and not after the generation calls.
    private = private_texts(work / "private_texts.jsonl")
    generated = write_texts(client, writers, plan, work / "generated.jsonl", workers)
    texts, drops, hits = prepare(generated, private, files)
    (work / "overlap.json").write_text(json.dumps(hits, indent=2, ensure_ascii=False), "utf-8")
    asks = track_asks(templates)
    records = label_pairs(client, panel, judge_b, writer_models, asks, texts,
                          work / "journal.jsonl", workers)  # fmt: skip
    return {
        "requests": len(plan),
        "requests_failed": sum(1 for r in read_jsonl(work / "generated.jsonl") if r.get("error")),
        "texts_written": len(generated),
        "private_texts": len(private),
        "private_sampled": (work / "private_texts.jsonl").is_file(),
        "dropped": dict(sorted(drops.items())),
        "held_out_templates": {cell: e["tid"] for cell, e in sorted(templates.items())},
        "tracks": write_candidates(texts, asks, records, out),
        "spent_usd": round(client.spent, 4),
    }


def dry_run_plan(templates: dict) -> dict:
    """What a run would ask for and what it would cost, from the plan alone."""
    plan = generation_plan()
    asks = track_asks(templates)
    calls, tracks = Counter(), {}
    for track in TRACKS:
        requests = [p for p in plan if p["track"] == track]
        texts = sum(p["n"] for p in requests)
        n_questions = len(asks[track])
        writers = Counter(p["writer"] for p in requests)
        private = PRIVATE_COUNT if track in PRIVATE_TRACKS else 0
        pairs = Counter()
        for p in requests:
            pairs["B+C" if p["writer"] == "A" else "A+C"] += p["n"] * n_questions
        if private:
            pairs["A+C"] += private * n_questions
        for judges, n in pairs.items():
            for judge in judges.split("+"):
                calls[judge] += n
        calls["G"] += writers["G"]
        calls["writer A"] += writers["A"]
        tracks[track] = {
            "requests": dict(writers),
            "generated_texts": texts,
            "private_texts": private,
            "questions": [a.task for a in asks[track]],
            "pairs_by_judges": dict(pairs),
            "pairs": sum(pairs.values()),
        }
    cost = sum(COST_PER_CALL[role] * n for role, n in calls.items())
    options = {a.task: len(outcomes(question_of(a.question))) for a in asks["sss"]}
    return {
        "tracks": tracks,
        "requests": len(plan),
        "texts": sum(t["generated_texts"] + t["private_texts"] for t in tracks.values()),
        "pairs": sum(t["pairs"] for t in tracks.values()),
        "calls_by_role": dict(calls),
        "cost_per_call_usd": COST_PER_CALL,
        "estimate_usd": round(cost, 2),
        "estimate_with_retries_usd": round(cost * RETRY_MARGIN, 2),
        "cap_usd": CAP_USD,
        "held_out_templates": {
            cell: {"tid": e["tid"], "options": options[cell]}
            for cell, e in sorted(templates.items())
        },  # fmt: skip
        "disjoint_from_training": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.generate")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=CANDIDATES)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--templates", type=Path, default=TEMPLATES)
    parser.add_argument("--build-report", type=Path, default=BUILD_REPORT)
    parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)
    parser.add_argument("--cap-usd", type=float, default=CAP_USD,
                        help="the cap over the whole step 8 ledger")  # fmt: skip
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; call nothing")
    parser.add_argument("--sample-private", action="store_true",
                        help="draw the corpus halves into --work; no paid call")  # fmt: skip
    args = parser.parse_args(argv)
    check_disjoint()
    templates = held_out_templates(args.templates, args.build_report)
    if args.dry_run:
        print(json.dumps(dry_run_plan(templates), indent=2, ensure_ascii=False))
        return 0
    args.work.mkdir(parents=True, exist_ok=True)
    if args.sample_private:
        rows = sample_private(args.data)
        path = args.work / "private_texts.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
        print(json.dumps(dict(Counter(r["track"] for r in rows)), indent=2))
        return 0
    judges, generator, _cheap = load_panel()
    client = make_client(args.ledger_dir, args.cap_usd)
    try:
        result = run(client, judges, generator, load_relabel_judge(), work=args.work,
                     out=args.out, files=built_files(args.data), templates=templates,
                     workers=args.workers)  # fmt: skip
    except STOPPING as error:
        print(f"stopped: {error}; restart on the same --work to resume", file=sys.stderr)
        return 1
    (args.work / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
