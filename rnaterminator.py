import os.path
import decimal
from statistics import mean
from rnaterminator_libs.hybrid_annotator import HybridAnnotator
from rnaterminator_libs.annotation_exporter import AnnotationExporter
from rnaterminator_libs.annotations_merger import AnnotationsMerger
import logging as logger
import argparse
import glob
import multiprocessing as mp
import pandas as pd
from itertools import product
from numpy import percentile
import pandas as pd

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
    parser.add_argument("--select_percentile", default=None, type=int,
                        help="Ignore coverage up to")
    parser.add_argument("--ignore_coverage_file", default=None, type=str,
                        help="Ignore coverage up to values mentioned in TSV file per lib")
    parser.add_argument("--annotation_type", required=True, type=str,
                        help="Specify a name for the annotation type")
    parser.add_argument("--stats_only", default=False, action='store_true',
                        help="Causes the program to generate peak stats only, Ignores parameters")
    parser.add_argument("--generate_peaks_percentiles", default=False, action='store_true',
                        help="Causes the program to consider ignore_coverage parameter as percentile value")
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
    peaks_counts = []
    peaks_cov = {}
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
                peaks_counts.extend(wig[1])
                peaks_cov.update(wig[2])
            all_locs.reset_index(inplace=True, drop=True)
            wig_pool.close()
        output[cond_name] = all_locs

    if args.generate_peaks_percentiles:
        peaks_cov = combine_peaks_cov(peaks_cov)
        for k in peaks_cov.keys():
            print(f"Generating coverage peaks percentiles for: {k}")
            peaks_cov[k] = [(x, percentile(peaks_cov[k], x)) for x in range(0, 101, 1)]
        peaks_perc_df = peaks_dict_to_df(peaks_cov)
        peaks_perc_df.to_csv(f"{os.path.dirname(args.gff_out)}/percentiles.tsv", index=False, sep="\t", header=True)
    peaks_counts_df = combine_peaks_counts(peaks_counts)
    peaks_counts_df.to_csv(f"{os.path.dirname(args.gff_out)}/peak_stats.tsv", index=False, sep="\t", header=True)
    if args.stats_only:
        return None
    # Export
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


def combine_peaks_counts(peaks_counts: list) -> pd.DataFrame:
    ret_df = pd.DataFrame(data=peaks_counts,
                          columns={"lib_type": str, "lib": str, "percentile": int, "score": float, "peaks_counts": int})
    ret_df.drop_duplicates(inplace=True)
    for i in ret_df.index:
        ret_df.at[i, "lib"] = ret_df.at[i, "lib"].rsplit("_", maxsplit=1)[0]
    ret_df = ret_df.groupby(["lib_type", "lib", "percentile", "score"], as_index=False).sum()
    return ret_df


def peaks_dict_to_df(peaks_cov: dict) -> pd.DataFrame:
    peaks_perc_df = pd.DataFrame(data=[(k, v) for k, v in peaks_cov.items()], columns=["lib", "percentile_info"])
    peaks_perc_df = peaks_perc_df.explode("percentile_info")
    peaks_perc_df.reset_index(inplace=True, drop=True)
    for i in peaks_perc_df.index:
        lib = peaks_perc_df.at[i, "lib"].split("_", maxsplit=3)
        peaks_perc_df.at[i, "lib_type"] = lib[0]
        peaks_perc_df.at[i, "lib"] = lib[3]
        peaks_perc_df.at[i, "percentile"] = peaks_perc_df.at[i, "percentile_info"][0]
        peaks_perc_df.at[i, "score"] = peaks_perc_df.at[i, "percentile_info"][1]
    peaks_perc_df.drop(["percentile_info"], inplace=True, axis=1)
    return peaks_perc_df


def process_single_wiggle(up_wig_path, down_wig_path, cond_name, refseq_paths, args):
    peak_annotator_obj = HybridAnnotator(up_wig_path=up_wig_path, down_wig_path=down_wig_path,
                                         cond_name=cond_name, refseq_paths=refseq_paths, args=args)
    peaks_df, peaks_counts, peaks_cov = peak_annotator_obj.predict()
    return peaks_df, peaks_counts, peaks_cov


def combine_peaks_cov(in_dict):
    out_dict = {}
    keys = set([x.replace("_forward", "").replace("_reverse", "") for x in in_dict.keys()])
    for k in keys:
        if k not in out_dict.keys():
            out_dict[k] = []
        for i in in_dict.keys():
            if k in i:
                out_dict[k].extend(in_dict[i])
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