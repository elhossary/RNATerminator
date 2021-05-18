import os.path

from rnaterminator_libs.hybrid_annotator import HybridAnnotator
from rnaterminator_libs.annotation_exporter import AnnotationExporter
from rnaterminator_libs.annotations_merger import AnnotationsMerger
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
                        help="Wiggle files (space separated), format path:cond:upstream/downstream:+/-")
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
    parser.add_argument("--stats_only", default=False, action='store_true',
                        help="Causes the program to generate peak stats only, Ignores parameters")
    parser.add_argument("--ignore_coverage", default=10, type=int,
                        help="Ignore coverage up to")
    parser.add_argument("--gff_out", required=True, type=str, help="Path to output GFF file")
    args = parser.parse_args()
    logger.info("Getting list of files")
    refseq_paths = []
    for rs_item in args.refseqs_in:
        for sub_rs_item in glob.glob(rs_item):
            refseq_paths.append(sub_rs_item)

    parsed_wig_paths_df = parse_wig_paths(args.wigs_in)
    conditions_names = parsed_wig_paths_df["condition_name"].unique().tolist()
    output = {}
    peaks_counts = {}
    for cond_name in conditions_names:
        cond_df = parsed_wig_paths_df[parsed_wig_paths_df["condition_name"] == cond_name]
        all_locs = pd.DataFrame()
        for orient in ["+", "-"]:
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
                all_locs = all_locs.append(wig[0], ignore_index=True)
                peaks_counts.update(wig[1])
            all_locs.reset_index(inplace=True, drop=True)
            wig_pool.close()
        output[cond_name] = all_locs
    peaks_counts_str = "Peak_distance_param\tIgnore_coverage_param\tLibrary_type\tLibrary_name\tPeak count\n"
    for k, v in sum_peaks(peaks_counts).items():
        lib_type = "rising" if "rising in k" else "falling"
        lib_name = k.split("_", maxsplit=1).replace('_', ' ')
        peaks_counts_str += f"{args.peak_distance}\t{args.ignore_coverage}\t{lib_type}\t{lib_name}\t{v}\n"

    # Export
    ## Stats
    with open(f"{os.path.dirname(args.gff_out)}/stats.tsv", "a") as f:
        f.write(peaks_counts_str)
    unique_lines = []
    with open(f"{os.path.dirname(args.gff_out)}/stats.tsv", "r") as f:
        for line in f.readlines():
            if line not in unique_lines:
                unique_lines.append(line)
    with open(f"{os.path.dirname(args.gff_out)}/stats.tsv", "w") as f:
        f.write("\t".join(unique_lines))
    if args.stats_only:
        return None
    ## GFF
    all_cond_data = pd.DataFrame()
    all_cond_names = "_".join(output.keys())
    for k, v in output.items():
        # Replicates unmerged
        AnnotationExporter(v, args).export(prefix=f"unmerged_{k}")
        # Replicates merged
        cond_merged_locs = AnnotationsMerger(v, args).merge()
        AnnotationExporter(cond_merged_locs, args).export(prefix=f"{k}")
        # all conditions merged
        all_cond_data = all_cond_data.append(v, ignore_index=True)
    all_cond_data.reset_index(inplace=True, drop=True)
    all_cond_data = AnnotationsMerger(all_cond_data, args).merge()
    AnnotationExporter(all_cond_data, args).export(prefix=all_cond_names)


def process_single_wiggle(up_wig_path, down_wig_path, cond_name, refseq_paths, args):
    peak_annotator_obj = HybridAnnotator(up_wig_path=up_wig_path, down_wig_path=down_wig_path,
                                         cond_name=cond_name, refseq_paths=refseq_paths, args=args)
    peaks_df, peaks_counts = peak_annotator_obj.predict()
    return peaks_df, peaks_counts


def sum_peaks(in_dict):
    out_dict = {}
    keys = set([x.replace("_forward", "").replace("_reverse", "") for x in in_dict.keys()])
    for k in keys:
        if k not in out_dict.keys():
            out_dict[k] = 0
        for i in in_dict.keys():
            if k in i:
                out_dict[k] += in_dict[i]
    return out_dict

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
