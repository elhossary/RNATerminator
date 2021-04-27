from rnaterminator_libs.hybrid_annotator import HybridAnnotator
from rnaterminator_libs.annotation_exporter import AnnotationExporter
from rnaterminator_libs.annotations_merger import ConditionsMerger
import logging as logger
import argparse
import glob
import multiprocessing as mp
import pandas as pd
from itertools import product

def main():
    #np.set_printoptions(suppress=True)
    logger.info("Reading user arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--refseqs_in", required=True, type=str, nargs="+",
                        help="Fasta files for the reference sequence (space separated)")
    parser.add_argument("--wigs_in", required=True, type=str, nargs="+",
                        help="Wiggle files (space separated), format path:cond:up/down:forward/reverse")
    parser.add_argument("--min_len", default=50, type=int,
                        help="Minimum allowed annotation length")
    parser.add_argument("--max_len", default=350, type=int,
                        help="Maximum allowed annotation length")
    parser.add_argument("--peak_distance", default=50, type=int,
                        help="Local maxima of derivative peaks for each library")
    parser.add_argument("--step_size", default=3, type=int,
                        help="Derivative range")
    parser.add_argument("--threads", default=1, type=int,
                        help="Parallel file processors")
    parser.add_argument("--ignore_coverage", default=10, type=int,
                        help="Ignore coverage up to")
    parser.add_argument("--annotation_type", required=True, type=str,
                        help="Specify a name for the annotation type")
    parser.add_argument("--Unmerge_replicates", default=False, action='store_true',
                        help="Stop mergeing overlapping annotations from replicates")
    parser.add_argument("--merge_conditions", default=False, action='store_true',
                        help="Merges overlapping annotations from different conditions")
    parser.add_argument("--gff_out", required=True, type=str, help="Path to output GFF file")
    args = parser.parse_args()
    logger.info("Getting list of files")
    refseq_paths = []
    for rs_item in args.refseqs_in:
        for sub_rs_item in glob.glob(rs_item):
            refseq_paths.append(sub_rs_item)

    parsed_wig_paths_df = parse_wig_paths(args.wigs_in)
    conditions_names = parsed_wig_paths_df["condition_name"].unique().tolist()
    for cond_name in conditions_names:
        cond_df = parsed_wig_paths_df[parsed_wig_paths_df["condition_name"] == cond_name]
        all_locs = pd.DataFrame()
        for orient in ["forward", "reverse"]:
            cond_orient_df = cond_df[cond_df["orientation"] == orient]
            up_wigs = cond_orient_df[cond_orient_df["wig_type"] == "upstream"]["path"].tolist()
            down_wigs = cond_orient_df[cond_orient_df["wig_type"] == "downstream"]["path"].tolist()
            combinations = product(up_wigs, down_wigs)
            wig_pool = mp.Pool(processes=args.threads)
            processes = []
            for comb in combinations:
                processes.append(wig_pool.apply_async(process_single_wiggle,
                                                      args=(comb[0], comb[1], cond_name, refseq_paths, args)))
            wiggles_processed = [p.get() for p in processes]
            for wig in wiggles_processed:
                all_locs = all_locs.append(wig, ignore_index=True)
            all_locs.reset_index(inplace=True, drop=True)
            wig_pool.close()

        cond_merged_locs = ConditionsMerger(all_locs, args).merge_replicates()
        print(cond_merged_locs.to_string())
        AnnotationExporter(cond_merged_locs, args).export(prefix=cond_name)


def process_single_wiggle(up_wig_path, down_wig_path, cond_name, refseq_paths, args):
    peak_annotator_obj = HybridAnnotator(up_wig_path=up_wig_path, down_wig_path=down_wig_path, cond_name=cond_name, refseq_paths=refseq_paths, args=args)
    peaks_df = peak_annotator_obj.predict()
    return peaks_df


def parse_wig_paths(wigs_paths):
    wig_info = [x.split(":") for x in wigs_paths]
    wig_info_extended = []
    for i, w in enumerate(wig_info):
        wig_paths = []
        for sub_w_item in glob.glob(w[0]):
            wig_paths.append(sub_w_item)
        wig_info[i].append(wig_paths)

    for w in wig_info:
        for wi in w[-1]:
            wig_info_extended.append([wi, w[1], w[2], w[3]])
    return pd.DataFrame(data=wig_info_extended, columns=["path", "condition_name", "wig_type", "orientation"])



main()
