from rnaterminator_libs.wiggle import Wiggle
import numpy as np
from functools import reduce
from Bio import SeqIO
import os
import logging as logger
from scipy.signal import find_peaks
import sys
import pandas as pd
import pybedtools as pybed
from io import StringIO


class HybridAnnotator:

    def __init__(self, up_wig_path, down_wig_path, refseq_paths, cond_name, args):
        self.refseq_paths = refseq_paths
        self.args = args
        self.arr_dict = {}
        self.cond_name = cond_name
        self.upstream_lib = ""
        self.downstream_lib = ""
        chrom_sizes = self.get_chrom_sizes(refseq_paths)
        up_wig_obj, down_wig_obj = Wiggle(up_wig_path, chrom_sizes, is_len_extended=True), \
                                   Wiggle(down_wig_path, chrom_sizes, is_len_extended=True)
        self.wig_orient = up_wig_obj.orientation
        self.transform_wiggle(up_wig_obj, down_wig_obj)
        del up_wig_obj, down_wig_obj
        logger.basicConfig(filename='example.log', encoding='utf-8', level=logger.DEBUG)
        logger.getLogger().addHandler(logger.StreamHandler(sys.stdout))

    def predict(self):
        out_df = pd.DataFrame()
        peaks_counts = {f"rising_{self.upstream_lib}": 0, f"falling_{self.downstream_lib}": 0}
        for seqid_key in self.arr_dict.keys():
            # Generate location
            tmp_df, r_peaks, f_peaks =\
                self.generate_locs(self.arr_dict[seqid_key],
                                   True if self.wig_orient == "r" else False,
                                   self.cond_name, seqid_key)
            print(f"\tPossible {tmp_df.shape[0]} positions for {self.cond_name} {self.wig_orient}")
            # Group overlaps and filter
            tmp_df = self.drop_overlaps(tmp_df, True if self.wig_orient == "r" else False)
            print(f"\t{tmp_df.shape[0]} valid positions for {self.cond_name} {self.wig_orient}")
            # append
            tmp_df["seqid"] = seqid_key
            out_df = out_df.append(tmp_df, ignore_index=True)
            peaks_counts[f"rising_{self.upstream_lib}"] += r_peaks
            peaks_counts[f"falling_{self.downstream_lib}"] += f_peaks

        out_df.reset_index(inplace=True, drop=True)
        return out_df, peaks_counts

    def generate_locs(self, coverage_array, is_reversed, cond_name, seqid):
        print(f"Generating all possible locations for: {cond_name} {seqid}{'R' if is_reversed else 'F'} ")
        if is_reversed:
            coverage_array = np.flipud(coverage_array)
        location_col = 0
        up_raw_coverage_col = 1
        rising_col = 2
        down_raw_coverage_col = 3
        falling_col = 4
        ## Find peaks
        rising_peaks, rising_peaks_props = find_peaks(coverage_array[:, rising_col],
                                                      height=(None, None),
                                                      prominence=(None, None),
                                                      distance=self.args.peak_distance)

        falling_peaks, falling_peaks_props = find_peaks(coverage_array[:, falling_col],
                                                        height=(None, None),
                                                        prominence=(None, None),
                                                        distance=self.args.peak_distance)

        rp_index_func = lambda x: np.where(rising_peaks == x)
        fp_index_func = lambda x: np.where(falling_peaks == x)
        ## Ignore low coverage
        rising_peaks_list = \
            [x for x in rising_peaks if coverage_array[x, up_raw_coverage_col] > self.args.ignore_coverage]
        falling_peaks_list = \
            [x for x in falling_peaks if coverage_array[x, down_raw_coverage_col] > self.args.ignore_coverage]
        falling_peaks_set = set(falling_peaks_list)
        strand = "-" if is_reversed else "+"
        possible_locs = []
        for rp in rising_peaks_list:
            rp_height = round(rising_peaks_props["peak_heights"][rp_index_func(rp)][0], 2)
            range_start = rp + self.args.min_len - 1
            range_end = rp + self.args.max_len - 1
            fp_range = set(range(range_start, range_end, 1))
            possible_fp = sorted(list(falling_peaks_set.intersection(fp_range)))
            if not possible_fp:
                continue
            possible_fp_heights = [falling_peaks_props["peak_heights"][fp_index_func(fp)][0] for fp in possible_fp]
            # possible_fp_prominences = [falling_peaks_props["prominences"][fp_index_func(fp)][0] for fp in possible_fp]
            counter = 0
            if len(possible_fp) > 1:
                diffs = np.diff(possible_fp_heights)
                if len(diffs) > 1:
                    for diff in diffs:
                        if diff < 0:
                            break
                        if possible_fp_heights[counter] / possible_fp_heights[counter + 1] > 0.10:
                            break
                        counter += 1
            fp_height = round(possible_fp_heights[counter], 2)
            lower_loc = int(coverage_array[rp, location_col])
            upper_loc = int(coverage_array[possible_fp[counter], location_col])
            if is_reversed:
                upper_loc, lower_loc = lower_loc, upper_loc
                fp_height, rp_height = rp_height, fp_height
            pos_len = upper_loc - lower_loc + 1
            possible_locs.append([lower_loc, upper_loc, strand, pos_len,
                                  self.upstream_lib, self.downstream_lib, cond_name,
                                  rp_height, fp_height])
        possible_locs_df = pd.DataFrame(data=possible_locs, columns=['start', 'end', 'strand', "position_length",
                                                                     "upstream_lib", "downstream_lib", "condition_name",
                                                                     "start_peak_height", "end_peak_height"])
        possible_locs_df["start"] = possible_locs_df["start"].astype(int)
        possible_locs_df["end"] = possible_locs_df["end"].astype(int)
        possible_locs_df["position_length"] = possible_locs_df["position_length"].astype(int)
        return self.drop_redundant_positions(possible_locs_df, is_reversed),\
               rising_peaks.shape[0], falling_peaks.shape[0]

    def drop_redundant_positions(self, df, is_reversed):
        sort_key = "end"
        height_node = "start_peak_height"
        if is_reversed:
            sort_key = "start"
            height_node = "end_peak_height"
        pos_keys = df[sort_key].unique().tolist()
        for key_pos in pos_keys:
            tmp_df = df[df[sort_key] == key_pos].copy()
            if tmp_df.shape[0] <= 1:
                continue
            df.drop(tmp_df.index.tolist(), inplace=True)
            tmp_df.sort_values([height_node], inplace=True, ascending=False)
            df = df.append(tmp_df.iloc[0], ignore_index=True)
        return df

    def drop_overlaps(self, df, is_reversed, df_size=0):
        print(f"Validating positions")
        df_size = df.shape[0]
        df = self.generate_grouping_column(df)
        sort_key = "group"
        pos_keys = df[sort_key].unique().tolist()
        for key_pos in pos_keys:
            tmp_df = df[df[sort_key] == key_pos].copy()
            if tmp_df.shape[0] <= 1:
                continue
            df.drop(tmp_df.index.tolist(), inplace=True)
            if is_reversed:
                tmp_df.sort_values(["end"], inplace=True, ascending=[False])
            else:
                tmp_df.sort_values(["start"], inplace=True, ascending=[True])
            tmp_df.drop(tmp_df.index[1], inplace=True, axis=0)
            df = df.append(tmp_df, ignore_index=True)
        if df_size == df.shape[0]:
            return df
        else:
            return self.drop_overlaps(df, is_reversed, df_size)

    def generate_grouping_column(self, df_in):
        print(f"Grouping overlapping annotations for: {self.cond_name}")
        df_in_subset = df_in[["start", "end"]].copy()
        df_in_subset["seqid"] = "chr1"
        df_in_subset = df_in_subset.reindex(columns=["seqid", "start", "end"])
        df_in_str = df_in_subset.to_csv(sep="\t", header=False, index=False)
        bed_locs = str(pybed.BedTool(df_in_str, from_string=True).sort().merge(d=-1))
        merged_bed_locs_df = pd.read_csv(StringIO(bed_locs), names=["seqid", "start", "end"], sep="\t")
        df_in["group"] = None
        group_counter = 1
        for i in merged_bed_locs_df.index:
            mask = df_in["start"].between(merged_bed_locs_df.at[i, "start"], merged_bed_locs_df.at[i, "end"])
            df_in.loc[mask, ["group"]] = group_counter
            group_counter += 1
        return df_in

    def transform_wiggle(self, up_wig_obj, down_wig_obj):
        wig_cols = ["variableStep_chrom", "location", "score"]
        up_wig_df = up_wig_obj.get_wiggle()
        down_wig_df = down_wig_obj.get_wiggle()
        self.upstream_lib = up_wig_df.iat[0, 1]
        self.downstream_lib = down_wig_df.iat[0, 1]
        print(f"Using replicates: {self.upstream_lib} vs. {self.downstream_lib}")
        up_wig_df = up_wig_df.loc[:, wig_cols]
        down_wig_df = down_wig_df.loc[:, wig_cols]
        up_wig_df["score"] = up_wig_df["score"].abs()
        down_wig_df["score"] = down_wig_df["score"].abs()

        merged_df = reduce(lambda x, y: pd.merge(x, y, on=["variableStep_chrom", "location"], how='left'),
                           [up_wig_df.loc[:, wig_cols],
                            up_wig_obj.to_step_height(self.args.step_size, "start_end").loc[:, wig_cols],
                            down_wig_df.loc[:, wig_cols],
                            down_wig_obj.to_step_height(self.args.step_size, "end_start").loc[:, wig_cols]])
        merged_df["location"] = merged_df["location"].astype(int)

        for seqid in merged_df["variableStep_chrom"].unique():
            tmp_merged = merged_df[merged_df["variableStep_chrom"] == seqid].drop("variableStep_chrom", axis=1).copy()
            ret_arr = np.absolute(tmp_merged.to_numpy(copy=True))
            self.arr_dict[seqid] = ret_arr

    def get_chrom_sizes(self, fasta_pathes):
        ret_list = []
        for fasta_path in fasta_pathes:
            logger.info(f"Parsing reference sequence: {os.path.basename(fasta_path)}")
            f_parsed = SeqIO.parse(fasta_path, "fasta")
            logger.info(f"Parsed  reference sequence: {os.path.basename(fasta_path)}")
            for seq_record in f_parsed:
                ret_list.append({"seqid": seq_record.id,
                                 "size": len(seq_record.seq),
                                 "fasta": os.path.basename(fasta_path)})
            logger.info(f"Chrom sizes added")
        return ret_list
