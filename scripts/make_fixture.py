#!/usr/bin/env python3
"""Create deterministic synthetic IGC-shaped data for pipeline smoke tests only.

This is not a real Icelandic corpus, a benchmark, or evidence of model quality.
The intentionally shuffled sentences are suitable for testing software plumbing.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path


SENTENCES = [
    "Í dag var bjart veður á Íslandi og fólk gekk niður að höfninni.",
    "Nýr skóli verður opnaður í bænum þegar framkvæmdum lýkur í haust.",
    "Bókasafnið býður gestum að lesa bækur og hlusta á íslenska tónlist.",
    "Rannsóknin fjallar um breytingar á náttúru landsins á síðustu árum.",
    "Börnin söfnuðust saman á leikvellinum eftir að kennslu lauk.",
    "Bændur segja að sumarið hafi verið gott og uppskeran sé mikil.",
    "Liðið vann leikinn eftir spennandi keppni á nýja íþróttavellinum.",
    "Gestir geta skoðað sýninguna í safninu alla daga vikunnar.",
    "Skipið kom til hafnar snemma morguns með vörur fyrir verslanir.",
    "Fjölskyldan fór í ferðalag um landið og heimsótti gamla vini.",
    "Á fundinum var rætt um samgöngur og þjónustu við íbúa.",
    "Höfundurinn las úr bók sinni og svaraði spurningum áheyrenda.",
    "Fuglar sátu við vatnið meðan sólin reis yfir fjöllunum.",
    "Sveitarfélagið hyggst bæta gönguleiðir og vernda græn svæði.",
    "Kennarar undirbúa verkefni um sögu tungumálsins fyrir nemendur.",
    "Vísindamenn mældu hitastig sjávar og skráðu niðurstöðurnar.",
    "Nýjar myndir sýna hvernig jökullinn hefur breyst með tímanum.",
    "Tónleikarnir fara fram á föstudag og aðgangur er ókeypis.",
    "Ferðamenn gengu eftir stígnum og nutu útsýnis yfir dalinn.",
    "Læknirinn sagði að hreyfing og nægur svefn væru mikilvæg.",
    "Íbúar hittust í félagsheimilinu til að ræða framtíð þorpsins.",
    "Starfsfólk vann saman að því að bæta aðgengi að upplýsingum.",
    "Nefndin mun kynna tillögur sínar fyrir lok mánaðarins.",
    "Íslenskar þjóðsögur hafa verið sagðar í margar kynslóðir.",
    "Húsið stendur við gömlu götuna þar sem áður var verslun.",
    "Nemendur lærðu að skrifa stuttar sögur og lesa þær upphátt.",
    "Sjómaðurinn lýsti ferðinni og sýndi myndir af norðurljósunum.",
    "Stjórnin hefur ákveðið að styrkja rannsóknir á endurnýjanlegri orku.",
    "Nýja brúin tengir saman hverfi sitt hvorum megin við ána.",
    "Listakonan notar liti og form til að segja frá bernsku sinni.",
    "Veðurstofan spáir rigningu við ströndina en sól á hálendinu.",
    "Verkefnið hófst í vetur og hefur fengið góðar viðtökur.",
    "Ráðstefnan sameinar fólk sem vinnur að menntun og menningu.",
    "Bókin fjallar um vináttu og daglegt líf í litlu sjávarþorpi.",
    "Markaðurinn er opinn um helgina og býður upp á íslenskan mat.",
    "Ungt fólk tekur þátt í sjálfboðastarfi við hreinsun fjörunnar.",
    "Í garðinum vaxa tré sem voru gróðursett fyrir mörgum árum.",
    "Sundlaugin verður lokuð í eina viku vegna viðhalds.",
    "Sérfræðingar ræða hvernig hægt sé að nýta gögn við skipulag.",
    "Gamlar ljósmyndir voru færðar í stafrænt safn fyrir almenning.",
]


def make_documents(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    documents = []
    for i in range(count):
        # Different combinations prevent near-dedup from merging the whole fixture.
        paragraphs = rng.sample(SENTENCES, 10)
        text = "\n\n".join(" ".join(paragraphs[j : j + 2]) for j in range(0, 10, 2))
        documents.append(
            {
                "document": text,
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-igc/{seed}/{i}")),
                "metadata": {
                    "source": f"synthetic_fixture_{i % 4}",
                    "publish_timestamp": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}T12:00:00",
                    "synthetic": True,
                    "purpose": "software_smoke_test_only",
                },
            }
        )
    # Exact and near duplicates exercise cleaning and group separation.
    for i in range(min(8, count)):
        duplicate = dict(documents[i])
        duplicate["uuid"] = f"duplicate-{i}"
        documents.append(duplicate)
    for i in range(min(4, count)):
        duplicate = dict(documents[i])
        duplicate["uuid"] = f"near-duplicate-{i}"
        duplicate["document"] = duplicate["document"] + "\n\nÞetta er viðbót."
        documents.append(duplicate)
    noise = [
        "",
        "12345 !@#$% 67890 <> " * 30,
        "Stutt.",
        "This is an intentionally foreign English test document. "
        "It exists only to exercise language filtering in a software test. "
        "It must not be counted as evidence of Icelandic training data. " * 3,
    ]
    for i, text in enumerate(noise):
        documents.append(
            {
                "document": text,
                "uuid": f"noise-{i}",
                "metadata": {
                    "source": "synthetic_noise",
                    "publish_timestamp": "2024-01-01T12:00:00",
                    "synthetic": True,
                },
            }
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fixture/raw"))
    parser.add_argument("--documents", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.documents < 80:
        parser.error("--documents must be at least 80 to exercise three splits")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "synthetic_fixture.jsonl"
    documents = make_documents(args.documents, args.seed)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    print(f"Wrote {len(documents)} synthetic records to {path}.")
    print("SMOKE TEST ONLY: not a real corpus and not evidence of language quality.")


if __name__ == "__main__":
    main()
