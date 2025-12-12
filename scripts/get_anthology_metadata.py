import json
import sys

ANTHOLOGY_PATH = "/home/recski/projects/acl-anthology/"
sys.path.append(f"{ANTHOLOGY_PATH}/python")
sys.path.append(f"{ANTHOLOGY_PATH}/bin")
from acl_anthology import Anthology
from create_hugo_data import paper_to_dict


def main():
    anthology = Anthology(datadir=f"{ANTHOLOGY_PATH}/data").load_all()

    papers = []
    for collection in anthology.collections.values():
        for volume in collection.volumes():
            volume_data = {
                "booktitle": volume.title.as_text(),
                "parent_volume_id": volume.full_id,
                "year": volume.year,
                "venue": volume.venue_ids,
            }
            for paper in volume.papers():
                data = paper_to_dict(paper)
                data.update(volume_data)
                papers.append(data)

    with open("../paper_data.json", "w") as of:
        for paper in papers:
            of.write(json.dumps(paper) + "\n")


if __name__ == "__main__":
    main()
