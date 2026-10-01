# Training data card

What the decision head trains on, where each part comes from, its licence, and
what is known to be wrong with it. Every row count is read from
results/step3/data_card.json, written by `python -m data.card` from the files
under data/built/: the first build of converted and generated rows, the rows
added in Run 2 and the conversation prompt-injection set added in Run 3. Licences, sources and
attributions are in data/MANIFEST.yaml, which `just check-licenses` enforces.

## Which run trains on what

Three training runs were scored on HakemBench, Runs 1 to 3; Run 3 is the
released model. Result files in the repository use the internal names r3, r4
and r5 for runs 1 to 3.

- **Two earlier development rounds and Run 1** (Run 1 gives the held-out
  generalisation result): the files of the first build, the first nine rows of
  the table below; the support held-out cells are never trained on. The mix
  takes at most 3,000 rows a task: 42,998 rows for Run 1.
- **Run 2**: the same files plus data/built/r4, the three rows of the table
  marked "Run 2": 55,870 mixed rows. Run 2 trains on the four support questions
  that Run 1 held out (Run 1 saw their three families only in other question
  types), so its held-out number is never quoted as generalisation.
- **Run 3**: Run 2's data plus the conversation set's train split (530 rows),
  in two variants with three random seeds each: one keeps half of Run 2's 721
  benign guardrail rows (360), the other leaves them out. The run the
  selection rule fixed in writing beforehand chose, one without those rows
  (bench/r5_select.py, results/step9/r5/selection.json), mixes 55,679 rows.
  The conversation set's validation file (its validation and test splits,
  220 rows) is the guardrail dev set: runs write predictions
  on it and nothing trains on it, not even through the prior weights.

Mixed row counts are each run's `mixed_rows` in `results/step4/round1/`, one file a run, named by the internal run name and seed (the released run is `r5b-base-s1.json`).

## Contents

| Source | Track | Question type | Labels | Train | Validation | Held-out tasks |
|---|---|---|---|---|---|---|
| MASSIVE 1.1, Turkish (converted) | niyet | choice, ten of the 59 intents a row | human | 10,890 | 1,978 | |
| OffensEval 2020 Turkish (converted) | moderasyon | yes or no | human | 28,537 | 3,169 | |
| MiDe22 (converted) | dogrulama | choice | human | 3,356 | 478 | |
| Generated decision rows on clips/mqa texts | sss | choice, yes or no, score | two judges | 12,957 | 2,834 | 2,311 |
| Constitutional Court individual applications (converted) | hukuk | choice, the right concerned | the court's database | 7,982 | 440 | |
| FACTurk fact-checks (converted) | dogrulama | choice, the verdict | five fact-checking organisations (Yalansavar's rows left out) | 13,095 | 1,470 | |
| Three Turkish prompt-injection sets (converted): Turkish Prompt-Injection 1K, Guardrail Hard Negatives (Turkish rows), Turkish Prompt Injections | guvenlik | yes or no | rule: the class each text was written, expanded or translated for | 1,791 | 224 | |
| WebFAQ Turkish (converted) | arama | yes or no, does the passage answer | rule: page structure | 17,989 | 2,000 | |
| Generated tracks: messages, fraud, grading, subject, court, checkable claim | spam, oltalama, egitim, hukuk, dogrulama | choice, yes or no, score | two judges | 10,250 | 3,378 | |
| Run 2: the four support questions on clips/mqa texts | sss | choice, yes or no, score | two labellers of one family, mean of both | 10,788 | 1,200 | |
| Run 2: benign messages to a product's assistant, written for training | guvenlik | yes or no | writer's intent, confirmed blind by a second labeller | 721 | 79 | |
| Run 2: comments written for training | moderasyon | yes or no | writer's intent, confirmed blind by a second labeller | 1,363 | 136 | |
| Run 3: Turkish Conversation Prompt-Injection Dataset (converted) | guvenlik | yes or no | rule: the class each text was written for | 530 | 220 (guardrail dev set, never trained on) | |

The first three converted sets were first used to compare backbones and then
became training data; the next four fill empty tracks. No source's test split
enters training; the conversation set's validation and test splits form the
guardrail dev set and are never trained on. The generated rows are described below.

The counts are after every removal of training rows that touch the support
development set (255 support questions with human answers, kept apart from the
test set) or HakemBench v1.0. The largest is on the legal-rights set:
506 rows removed before the freeze and 57 at the freeze, whose court summaries
share the court's formulas with the benchmark's legal items.

## The generated rows

**Texts.** 6,000 question and answer pairs sampled from clips/mqa (CC0 1.0 on
its packaging): half from company FAQ pages, half from community question
sites, from 1,784 sites, at most three pairs per FAQ site and 200 per community
site. Dropped at sampling: complaint sites, gambling and betting
pages, pages with markup or placeholders, link-only answers, text that is not
Turkish, mis-decoded text; 511 of the 6,000 were dropped after the sample was
drawn, when the filter was tightened. Promotion drops 610 labelled rows that
later rules catch: a betting-brand rule, and pages whose question is a
patient's message cut at a fixed length and whose answer repeats it with a
pointer to a doctor's page, SEO pages that answer with "call us", and slot-game
pages. Masked: e-mail addresses, links, phone
numbers, IBANs, national identity numbers, user handles, and names next to a
greeting or before "Bey" or "Hanım".

**Questions.** Written by various LLMs as templates, one question and option
set applied to many texts, in eight families (topic, intent, answer adequacy,
urgency, information type, human handover, sensitivity, audience) and three
types. Of 300 requests, 207 templates passed the rule checks and near-duplicate
test, 134 passed a one-call review, and 96 contribute rows after screening on
their first 16 texts. Four family and type cells are held out whole for the
generalisation test: sensitivity yes-or-no, handover choice, adequacy choice
and adequacy score (14 templates, 2,311 rows, evaluation texts only).

**Labels.** Each row's target is the mean of two judges' probability vectors,
each read from the judge's first-token log-probabilities or stated
probabilities with the options in rotated order. The judges agree on the
top answer on 81 percent of training rows and 77 percent of held-out rows.
Within a template, the most common answer is capped at 65 percent of training
rows for yes-or-no and 50 percent for choice and score; the evaluation splits
are not capped and keep the real prior.

**Mined rows.** 1,599 training rows carry the recipe `build-v1-mined`. Their
templates had a rare answer (urgency, handover, sensitivity), so one judge
read further training texts and only those it leaned towards the rare answer
on went to both judges. That judge's vote is half of these rows' target, so
their targets lean towards the rare answer on borderline texts. The
validation split has none of these rows.

**Decontamination.** The n-gram and near-duplicate check runs against 70
registered evaluation sets, 328,398 items: the test splits of the backbone
comparison, Cetvel, TrGLUE and TabiBench. In the first build it flagged 3 of
the 18,871 generated rows, 13 and 3 MASSIVE rows, 33 and 5 OffensEval rows, and 188 and
28 MiDe22 rows (train and validation), the last near-duplicates of the MiDe22
test fifth carved by the converter, as expected; and of the sets added later, 36 FACTurk rows (28 of
them TrClaim19 claims, a Cetvel set), one prompt-injection row and three WebFAQ
rows. Every flagged row is removed before training
(results/step3/decontam/, `data.decontam.apply`). The conversation set went
through the same check and none of its 750 rows was flagged.

## The rows added in Run 2

**Support.** 2,697 training and 300 validation texts: existing clips/mqa
question-and-answer texts from the support text pools (the support development
set's texts left out), labelled anew with the benchmark's four support questions in their
written wording: sensitivity (yes or no),
needs a person (choice), answer adequacy (choice, and again as a score). Two
labellers, AI models of one family run as local agents, give a
distribution per question, one in the benchmark's option order and one with
the choice options shuffled; the target is their mean, so a disagreement stays
a soft target. They pick the same answer on 95 (sensitivity), 89 (needs a
person), 88 (adequacy, choice) and 87 percent (adequacy, score) of texts.

**Guardrails and moderation.** New texts written by one of the two models in
twenty product contexts and twenty comment forms none of the benchmark's
writer models used, labelled blind by the other; a text is kept only where the
labeller agrees with the intent it was written for, and its target is the mean
of that intent and the labeller's distribution. The guardrail texts are benign
only: every request for attack messages was stopped by the writing model's
safety filter. 800 guardrail texts (721 train, 79 validation) and 1,499
of 1,500 comments (899 offensive, 600 clean) were kept. Near duplicates are
removed, and so is every text that touches a HakemBench v1.0 item, a probe or a
support development set text.

## The conversation prompt-injection set

Turkish Conversation Prompt-Injection Dataset, version 1.0.2, by Enes Deniz (CC
BY 4.0). Its rows were written with a model and curated by its author; its card
does not claim independent human annotation. Its train split trains as task
prompt_injection-conv (530 rows after the overlap checks of
results/step9/r5/overlap.json), with the prompt-injection question wording of
the three converted sets. Its validation and test splits (20 and 30 attacks, 80
and 90 benign) form the guardrail dev set on which the released run was chosen.

## Known limits

- The judges are LLMs and make mistakes a reader would not: in a read sample,
  an airport timetable was labelled as a digital topic with full confidence.
- Community questions were read by title only; the question body, where
  people describe their situation, was not sampled. Community texts are
  therefore shorter than the originals.
- Urgency is thin: 108 training rows. Handover has 1,885 in the first build, and
  Run 2 adds 2,697 handover choice rows.
- Three community sites give 200 pairs each: a health forum, a mothers' forum and a
  general question site.
- clips/mqa licenses its packaging under CC0 and disclaims the pages' text
  (data/MANIFEST.yaml). The training rows built from it are not published.
- FACTurk is 93 percent "false or misleading", the mix of the archives it
  comes from; it teaches the prior more than the distinction.
- Every prompt-injection row is rule-labelled: expanded from templates,
  machine-translated, or written with a model (the conversation set), and
  labelled by the class each text was written for. WebFAQ's relevance comes
  from page structure. None of these labels was checked by a person.
- Before Run 3, the only guardrail training texts in the natural register of
  a message to a product's assistant were Run 2's benign rows, and every
  training attack was a template or a short translated prompt. This is the
  likely cause of Run 2's drop in attack recall; Run 3 adds the conversation
  set's attacks in that register.
- Run 2's support labels come from the family that decided the benchmark's
  support gold under the same written rule, so the model's support score on AI gold
  is partly agreement with its labeller; the human answers on the support
  questions of the human check are a check outside that family.
- Run 2's comments share surface style with the benchmark's generated moderation
  items. A character n-gram classifier trained on them reaches balanced
  accuracy 0.926 on the generated items against 0.818 for one trained on the
  old rows, and 0.401 against 0.619 on web-sourced moderation texts. The
  open release holds only the generated moderation items, so the model's moderation result may read
  higher than it should.
- Spam, phishing and education are generated texts only: messages written
  by one model to intended kinds, with fictional names and masked links, and
  students' answers of intended quality; we found no human-labelled Turkish set
  licensed for this use. The message tasks are judged by two judges that did not
  write them.
- The generated legal question asks which court or authority deals with a lay
  description; its scenes were checked against procedure, and mandatory
  mediation, which comes before several of these courts, is not asked.
- OffensEval and MiDe22 are tweets. They are training data only; the
  project keeps tweets out of the benchmark.
- Held-out score rows are few, so the generalisation test on score questions
  is weak.
- The validation splits of MASSIVE and MiDe22 lost their flagged rows, so
  numbers on them are not comparable with those of the first head test
  (results/step4/slice/).
