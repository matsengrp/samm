import json
from motif_feature_generator import MotifFeatureGenerator

def process_model_json(fname):
    """
    Ex:
    {
        'motif_length': '3',
        'distances_from_motif_start': '-2,-1,0',
        'motifs_to_keep': '',
    }
    """
    feat_gens = []
    feats_to_remove = []
    flanks = {}
    with open(fname, 'r') as f:
        models = []
        lefts = []
        rights = []
        lens = []
        for model in json.load(f):
            model['motif_length'] = int(model['motif_length'])
            lens.append(model['motif_length'])
            model['distances_from_motif_start'] = [int(dist) for dist in model['distances_from_motif_start'].split(',')]
            model['motifs_to_keep'] = [motif for motif in model['motifs_to_keep'].split(',') if motif]
            models.append(model)
            # get flank_len_offset, left_update, left_flank_len, etc.
            motif_len = model['motif_length']
            for distance_to_start in model['distances_from_motif_start']:
                lefts.append(distance_to_start)
                rights.append(motif_len - 1 + distance_to_start)

        flanks['left_flank_len'] = -min(min(lefts), 0)
        flanks['right_flank_len'] = max(max(rights), 0)
        flanks['max_motif_len'] = max(lens)

        for model in models:
            motif_len = model['motif_length']
            for distance_to_start in model['distances_from_motif_start']:
                feat_gens.append(
                    MotifFeatureGenerator(
                        motif_len=motif_len,
                        distance_to_start_of_motif=distance_to_start,
                        flank_len_offset=flanks['left_flank_len'] + distance_to_start,
                    )
                )
                if model['motifs_to_keep']:
                    feats_to_remove += [feat_tuple for feat_tuple in feat_gens[-1].feature_info_list if feat_tuple[0] not in model['motifs_to_keep']]

    return feat_gens, feats_to_remove, flanks

