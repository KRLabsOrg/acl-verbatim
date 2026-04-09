import argparse
import json
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract paper metadata from a local ACL Anthology repository."
    )
    parser.add_argument(
        "--anthology-path",
        required=True,
        help="Path to the root of a local clone of the acl-anthology repository.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output JSONL file (one JSON object per line).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    sys.path.append(f"{args.anthology_path}/python")
    sys.path.append(f"{args.anthology_path}/bin")
    from acl_anthology import Anthology
    from create_hugo_data import paper_to_dict

    anthology = Anthology(datadir=f"{args.anthology_path}/data").load_all()

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

    with open(args.output_file, "w") as of:
        for paper in papers:
            of.write(json.dumps(paper) + "\n")


if __name__ == "__main__":
    main()
