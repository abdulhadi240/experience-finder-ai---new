# trip_drivers — Schema Reference

Extracted automatically by the trip planning agent from **user messages only**.  
Never sourced from assistant text. Attached to every trip plan response as `trip_drivers[]`.

---

## Example payload

```json
{
  "trip_drivers": [
    {
      "theme": "adventure",
      "priority": "exclusive",
      "score": 0.95,
      "confidence": 0.95,
      "user_evidence": "best places to do paragliding in the Alps",
      "destination_driver": true,
      "specific_activity": "paragliding",
      "desired_frequency": "multiple_days"
    },
    {
      "theme": "food",
      "priority": "preferred",
      "score": 0.30,
      "confidence": 0.75,
      "user_evidence": "I enjoy good restaurants",
      "destination_driver": false,
      "specific_activity": null,
      "desired_frequency": "daily"
    }
  ]
}
```

---

## Fields

### `theme` — `string`

Broad category bucket. Fixed vocabulary:

| Value | Covers |
|---|---|
| `adventure` | paragliding, rafting, bungee, zip-lining, extreme sports |
| `fishing` | sport fishing, fly fishing, deep-sea, shore fishing |
| `beaches` | beach days, coastal relaxation, swimming |
| `food` | restaurants, markets, street food, culinary experiences |
| `hiking` | trails, trekking, mountain walks |
| `culture` | museums, heritage sites, art galleries, local customs |
| `nightlife` | bars, clubs, live music, evening entertainment |
| `romance` | couples experiences, sunset dining, intimate stays |
| `wellness` | spa, yoga, meditation, thermal baths |
| `family` | kid-friendly attractions, theme parks, zoos |
| `wine` | wineries, vineyards, tastings |
| `surfing` | surfing, board sports |
| `diving` | scuba, snorkelling, freediving |
| `skiing` | ski resorts, snowboarding, winter sports |
| `photography` | viewpoints, landmarks, golden hour spots |
| `luxury` | premium hotels, fine dining, private tours |
| `budget` | free attractions, hostels, budget travel |
| `wildlife` | safaris, reserves, birdwatching, marine sanctuaries |
| `history` | archaeological sites, ruins, heritage walks |

---

### `priority` — `enum`

How central this theme is to the trip. **5 values, ordered by intensity:**

| Value | Score range | User signals |
|---|---|---|
| `"incidental"` | 0.10–0.20 | "maybe if there's time", "could be fun" |
| `"preferred"` | 0.25–0.40 | "I like / enjoy / would love some X" |
| `"important"` | 0.45–0.65 | "I want X", "make sure there's X", "include X" |
| `"primary"` | 0.70–0.90 | "mainly for X", "mostly about X", "trip is around X" |
| `"exclusive"` | 0.90–1.00 | "basically for X", "whole point is X", "where best for X?" |

> Use `priority` for human-readable logic and UI labels.  
> Use `score` for numeric weighting in algorithms.

---

### `score` — `float` (0.0–1.0)

Numeric representation of `priority`. Use this for weighting and ranking:

| Priority | Typical score |
|---|---|
| incidental | ≈ 0.15 |
| preferred | ≈ 0.30 |
| important | ≈ 0.55 |
| primary | ≈ 0.80 |
| exclusive | ≈ 0.95 |

---

### `confidence` — `float` (0.0–1.0)

How certain the model is about the classification.

| Range | Meaning |
|---|---|
| 0.5–0.65 | Ambiguous phrasing — inferred, not stated |
| 0.70–0.85 | Reasonably clear statement |
| 0.90–0.95 | Explicit, unambiguous statement from the user |

Low confidence (`< 0.65`) means the driver exists but the priority level may be off by one step.

---

### `user_evidence` — `string`

The exact phrase from the user's message that triggered this driver. Never paraphrased. Use for debugging, display, or explainability ("We suggested fishing because you said: *'mostly for fly fishing'*").

---

### `destination_driver` — `boolean`

Whether this theme is **why the user chose (or is searching for) the destination.**

| Value | Scenario |
|---|---|
| `true` | "Where should I go for the best surfing?" — theme is driving destination selection |
| `true` | "We chose Costa Rica because of the fishing" — destination picked due to theme |
| `false` | Destination already chosen; theme is an add-on ("going to Japan, want good food") |
| `false` | Companion preference ("my wife also wants beaches") |

When `destination_driver: true`, this theme should influence **destination ranking and selection**.  
When `false`, it shapes the **itinerary** but not which place to go.

---

### `specific_activity` — `string | null`

The exact activity named by the user when it is **more specific than the theme**.

| User says | `theme` | `specific_activity` |
|---|---|---|
| "paragliding" | `adventure` | `"paragliding"` |
| "fly fishing" | `fishing` | `"fly fishing"` |
| "white water rafting" | `adventure` | `"white water rafting"` |
| "cave diving" | `diving` | `"cave diving"` |
| "surfing trip" | `surfing` | `null` — same word as theme |
| "fishing trip" | `fishing` | `null` — same word as theme |
| "water activities" | `adventure` | `null` — too vague, treat as null |

> **Backend note:** if `specific_activity` is a vague catch-all like `"water activities"` or `"outdoor activities"`, treat it the same as `null` — use `theme` only.

---

### `desired_frequency` — `string | null`

How often the user wants this activity during the trip.

| Value | Meaning |
|---|---|
| `"once"` | A single occurrence is enough ("one wine tasting") |
| `"multiple_days"` | Across several days but not every day |
| `"daily"` | Expected every day ("good restaurants daily") |
| `"throughout"` | Woven into every part of the trip ("culture throughout") |
| `null` | Not stated |


