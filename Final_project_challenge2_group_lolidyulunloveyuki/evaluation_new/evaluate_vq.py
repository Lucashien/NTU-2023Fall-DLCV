import argparse
import json

import tqdm
from metrics import compute_visual_query_metrics
from structures import ResponseTrack, BBox
import pdb

def validate_model_predictions(predictions, gt_annotations):
    assert len(predictions) == len(gt_annotations)
    n_samples = 0
    for uid, c in gt_annotations.items():
        for a in c["annotations"]:
            for _, q in a["query_sets"].items():
                if q["is_valid"]:
                    n_samples += 1

    pbar = tqdm.tqdm(total=n_samples, desc="Validating user predictions")
    for annot_uid, pred_uid, clip_annots, clip_preds in zip(gt_annotations, predictions, gt_annotations.values(), predictions.values()):
        assert annot_uid == pred_uid
        assert type(clip_preds["predictions"]) == type([])
        assert len(clip_preds["predictions"]) == len(clip_annots["annotations"])
        for clip_annot, clip_pred in zip(
            clip_annots["annotations"], clip_preds["predictions"]
        ):
            assert type(clip_pred) == type({})
            assert "query_sets" in clip_pred
            valid_query_set_annots = {
                k: v for k, v in clip_annot["query_sets"].items() if v["is_valid"]
            }
            valid_query_set_preds = {
                k: v
                for k, v in clip_pred["query_sets"].items()
                if clip_annot["query_sets"][k]["is_valid"]
            }
            assert set(list(valid_query_set_preds.keys())) == set(
                list(valid_query_set_annots.keys())
            )
            for qset_id, qset in clip_pred["query_sets"].items():
                assert type(qset) == type({})
                for key in ["bboxes", "score"]:
                    assert key in qset
                pbar.update()


def evaluate(gt_file, pred_file):
    print("Starting Evaluation.....")

    with open(gt_file, "r") as fp:
        gt_annotations = json.load(fp)
    with open(pred_file, "r") as fp:
        model_predictions = json.load(fp)

    # Validate model predictions
    validate_model_predictions(model_predictions, gt_annotations)

    # Convert test annotations, model predictions to the correct format
    predicted_response_tracks = []
    annotated_response_tracks = []
    visual_crop_boxes = []
    for annos_uids, preds_uids, clip_annos, clip_preds in zip(gt_annotations, model_predictions, gt_annotations.values(), model_predictions.values()):
        for clip_anno, clip_pred in zip(
            clip_annos["annotations"], clip_preds["predictions"]
        ):
            qset_ids = list(clip_anno["query_sets"].keys())
            for qset_id in qset_ids:
                if not clip_anno["query_sets"][qset_id]["is_valid"]:
                    continue
                q_anno = clip_anno["query_sets"][qset_id]
                q_pred = clip_pred["query_sets"][qset_id]
                rt_pred = ResponseTrack.from_json(q_pred)
                rt_anno = []
                for rf in q_anno["response_track"]:
                    rt_anno.append(
                        BBox(
                            rf["frame_number"],
                            rf["x"],
                            rf["y"],
                            rf["x"] + rf["width"],
                            rf["y"] + rf["height"],
                        )
                    )
                rt_anno = ResponseTrack(rt_anno)
                vc = q_anno["visual_crop"]
                vc_bbox = BBox(
                    vc["frame_number"],
                    vc["x"],
                    vc["y"],
                    vc["x"] + vc["width"],
                    vc["y"] + vc["height"],
                )
                predicted_response_tracks.append([rt_pred])
                annotated_response_tracks.append(rt_anno)
                visual_crop_boxes.append(vc_bbox)

    # Perform evaluation
    pair_metrics = compute_visual_query_metrics(
        predicted_response_tracks,
        annotated_response_tracks,
        visual_crop_boxes,
    )

    print("Evaluating VQ2D performance")
    for pair_name, metrics in pair_metrics.items():
        print("-" * 20)
        print(pair_name)
        print("-" * 20)
        metrics = {
            "stAP @ IoU=0.25": metrics[
                "SpatioTemporal AP              @ IoU=0.25     "
            ],
        }
        for k, v in metrics.items():
            print(f"{k:<20s} | {v:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True, type=str)
    parser.add_argument("--pred-file", required=True, type=str)

    args = parser.parse_args()

    evaluate(args.gt_file, args.pred_file)
