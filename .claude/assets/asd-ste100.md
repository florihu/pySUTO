# ASD-STE100 Quick Reference

This is a text copy of the "Simplified Technical English: overview" sheet in [Andrej Karpathy's post](https://x.com/karpathy/status/2105819303471976479). The image is at [assets/asd-ste100-overview.png](../../../assets/asd-ste100-overview.png). Owner of the specification: ASD. Source: [asd-ste100.org](https://www.asd-ste100.org).

The spec is stringent. Aim for about 80% of the way to it.

## Document structure

ASD-STE100 has two parts.

- **Part 1: Writing rules.** Section 1 Words, 2 Noun clusters, 3 Verbs, 4 Sentences, 5 Procedures, 6 Descriptive writing, 7 Safety instructions, 8 Punctuation and word counts, 9 Writing practices. Rules are numbered by section, for example Rule 5.1 belongs to Section 5, Procedures.
- **Part 2: Dictionary.**
  - Approved words: keyword in UPPERCASE, one meaning.
  - Unapproved words: lowercase, with an approved alternative.
  - Technical names: company terms, defined categories.
  - Technical verbs: company verbs, defined categories.
  - About 900 approved words. Each word has one part of speech and one meaning.

## Writing rule limits

| Item | Maximum |
|------|---------|
| Procedural sentence | 20 words |
| Descriptive sentence | 25 words |
| Descriptive paragraph | 6 sentences |
| Noun cluster | 3 words |
| Instructions in each sentence | 1 (except simultaneous actions) |

- Use the same word for the same thing every time.
- Do not leave out words like "the", "a", and "this".
- Use the active voice in procedures.
- Use vertical lists for complex text.
- Write one topic in each paragraph.

## Verb forms

| Form | Example | Status |
|------|---------|--------|
| Command (imperative) | Close the valve. | Approved |
| Simple present | The valve closes. | Approved |
| Simple past | The valve closed. | Approved |
| Simple future | The valve will close. | Approved |
| Infinitive | Turn the knob to close it. | Approved |
| Past participle (as adjective) | The closed valve | Approved |
| Progressive (-ing) | The valve is closing. | Not approved |
| Perfect | The valve has closed. | Not approved |
| Passive, in procedures | The valve must be closed. | Not approved |

Use the -ing form only inside a technical name, for example "landing gear". Descriptive text can use the passive only when it is necessary.

## Dictionary entries

| Word (part of speech) | Status | Meaning or alternative | Approved example | Not approved |
|-----------------------|--------|------------------------|------------------|--------------|
| CLOSE (v) | Approved | To move together; to stop flow | CLOSE the valve. | - |
| close (adj) | Not approved | NEAR | Put the tool NEAR the panel. | Put the tool close to the panel. |
| commence (v) | Not approved | START | START the pump. | Commence pumping. |
| ensure (v) | Not approved | MAKE SURE | MAKE SURE THAT the switch is off. | Ensure the switch is off. |
| prior to (prep) | Not approved | BEFORE | BEFORE you start the engine, | Prior to starting the engine, |
| replenish (v) | Not approved | FILL | FILL the reservoir. | Replenish the reservoir. |
| utilize (v) | Not approved | USE | USE a torque wrench. | Utilize a torque wrench. |
| approximately (adv) | Not approved | ABOUT | Wait for ABOUT 10 minutes. | Wait approximately 10 minutes. |
| TEST (v) | Approved | To find if it operates correctly | TEST the circuit. | - |
| in order to | Not approved | TO | Remove the panel TO get access. | Remove the panel in order to access. |

Convention: approved words are shown in UPPERCASE, unapproved words in lowercase. An approved word keeps only its listed meaning and part of speech. "Close" is a verb only, never an adjective meaning "near".

## Anatomy of STE sentences

Original text, not STE (unapproved words: imperative, ensures, replenished, prior to, commencing):

> It is imperative that the operator ensures the hydraulic reservoir is replenished prior to commencing operation.

**1. Procedural sentence** (the same instruction, rewritten; 13 words, limit 20):

> Make sure that the hydraulic reservoir is full before you start the operation.

- "Make sure": approved verb, command form.
- "hydraulic reservoir": technical name, 2-word noun cluster (max 3).
- "full", not "replenished". "before", not "prior to". "start", not "commencing".
- "you start": active voice.

**2. Safety instruction:**

> WARNING: Do not touch the brake unit until it is cool. Hot parts can cause injury.

- First a clear, simple command. Then the risk.
- WARNING = risk of injury. CAUTION = risk of damage.

**3. Descriptive sentence** (12 words, limit 25):

> The pump supplies fuel to the engine when the switch is on.

- "supplies": simple present tense.
- Max 6 sentences per paragraph, one topic per paragraph.

## History

- 1979: AECMA starts work on controlled English for airlines.
- 1986: First AECMA Simplified English Guide published.
- 2005: Becomes ASD-STE100 after AECMA joins ASD in 2004.
- Now: Free download, revised by the ASD STEMG (Simplified Technical English Maintenance Group). Also used in defence, rail, and industry.
