"""Format-only fixture at the supplied dataset's scale; no original images."""
from collections import Counter
import csv
import json
from pathlib import Path

from ai6103.autodata import find_layout
from ai6103.data import discover,validate_rows


def test_observed_assignment_format_full_counts_labels_and_known_overlap(tmp_path):
    contract=json.loads((Path(__file__).parent/"fixtures/assignment_format.json").read_text())
    root=tmp_path/"data"
    samples=contract["sample_breed_names"]
    breeds=samples+[f"synthetic_breed_{i}" for i in range(120-len(samples))]
    expected={}
    identity_index=0
    shared_bytes=None
    for split,details in contract["splits"].items():
        folder=root/split/"images"
        folder.mkdir(parents=True)
        # Only the extrema and total were observed. This middle distribution is
        # deliberately synthetic; it is not an inferred empirical distribution.
        low,high=details["class_size_min"],details["class_size_max"]
        remaining=details["label_rows"]-low-high
        quotient,remainder=divmod(remaining,118)
        counts=[low,high]+[quotient+(i<remainder) for i in range(118)]
        assert min(counts)==low and max(counts)==high
        records=[]
        for breed_index,(breed,count) in enumerate(zip(breeds,counts)):
            for index in range(count):
                identity=f"{identity_index:032x}"
                identity_index+=1
                content=b"FORMAT-ONLY FIXTURE; NOT A JPEG: "+identity.encode()
                if split=="train" and breed_index==0 and index==0:
                    shared_bytes=content
                if split=="val" and breed_index==0 and index==0:
                    content=shared_bytes
                (folder/f"{identity}.jpg").write_bytes(content)
                records.append({"id":identity,"breed":breed})
                expected[str((folder/f"{identity}.jpg").resolve())]=(split,breed)
        # CSV order intentionally differs from image filename order.
        with (root/f"{split}_labels.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=contract["columns"])
            writer.writeheader()
            writer.writerows(reversed(records))
    detected,manifest,resplit=find_layout(root,tmp_path/"work")
    assert detected==root and manifest is not None and not resplit
    rows,_=discover(detected,manifest)
    assert len(rows)==10222
    assert {r["path"]:(r["split"],r["label"]) for r in rows}==expected
    audit={}
    classes,counts=validate_rows(rows,strict=True,duplicate_policy="report",audit=audit)
    assert len(classes)==120
    assert counts=={"train":7155,"val":1533,"test":1534}
    for split,details in contract["splits"].items():
        sizes=Counter(r["label"] for r in rows if r["split"]==split)
        assert len(sizes)==120
        assert (min(sizes.values()),max(sizes.values()))==(details["class_size_min"],details["class_size_max"])
    assert audit["overlap_images"]=={"train_val":1,"train_test":0,"val_test":0}
    assert audit["cross_split_duplicate_groups"]==1
