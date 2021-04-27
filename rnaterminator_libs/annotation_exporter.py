import os
import pandas as pd


class AnnotationExporter:
    
    def __init__(self, df: pd.DataFrame, args):
        self.df = df
        self.args = args
    
    def export(self, prefix="") -> None:
        if self.df.shape[0] == 0:
            return None
        out_path = os.path.abspath(self.args.gff_out)
        if prefix != "":
            out_path = self._inject_prefix(self.args.gff_out, prefix)

        strand_letter_func = lambda x: "F" if x == "+" else "R"
        col_names = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
        peaks_gff_df = pd.DataFrame(columns=col_names)
        #self.df = self.df.apply(lambda row: self._join_lists(row), axis=0)
        for i in self.df.index:
            seqid = self.df.at[i, 'seqid']
            strand = strand_letter_func(self.df.at[i, 'strand'])
            attr = f"ID={seqid}_{strand}_{i}" \
                   f";Name={seqid}_{strand}_{self.args.annotation_type}_{i}" \
                   f";conditions={self.df.at[i, 'condition_name']}" \
                   f";start_peak_height={self.df.at[i, 'start_peak_height']}"\
                   f";end_peak_height={self.df.at[i, 'end_peak_height']}" \
                   f";upstream_libs={self.df.at[i, 'upstream_lib']}" \
                   f";downstream_libs={self.df.at[i, 'downstream_lib']}" \
                   f";seq_len={str(self.df.at[i, 'position_length'])}".replace("__", "_")
            peaks_gff_df = peaks_gff_df.append({"seqid": self.df.at[i, "seqid"],
                                                "source": "RNATerminator",
                                                "type": self.args.annotation_type,
                                                "start": self.df.at[i, "start"],
                                                "end": self.df.at[i, "end"],
                                                "score": ".",
                                                "strand": self.df.at[i, "strand"],
                                                "phase": ".",
                                                "attributes": attr}, ignore_index=True)
        """
        print("Filtering by length")
        peaks_gff_df["len"] = peaks_gff_df["end"] - peaks_gff_df["start"] + 1
        len_range = range(self.args.min_len, self.args.max_len + 1, 1)
        print(f"\t- Before length filtering: {peaks_gff_df.shape[0]}")
        peaks_gff_df = peaks_gff_df[peaks_gff_df["len"].isin(len_range)]
        print(f"\t- After length filtering: {peaks_gff_df.shape[0]}")
        peaks_stats = peaks_gff_df["len"].describe(include='all')
        peaks_gff_df.drop("len", axis=1, inplace=True)
        """
        print("Sorting annotated peaks")
        peaks_gff_df.sort_values(["seqid", "start", "end"], inplace=True)
        print(f"Total {peaks_gff_df.shape[0]} peaks predicted, exporting to GFF")
        # f"median and average lengths: {peaks_stats['median']} and {peaks_stats['mean']}, exporting to GFF")
        peaks_gff_df.to_csv(out_path, sep="\t", header=False, index=False)


    @staticmethod
    def _join_lists(row: pd.Series) -> pd.Series:
        for k in row.keys():
            if type(row[k]) is list:
                row[k] = ','.join(str(row[k]))
        return row

    @staticmethod
    def _inject_prefix(in_path, prefix):
        dir = os.path.dirname(in_path)
        fname = os.path.basename(in_path)
        out_path = f"{dir}/{prefix}_{fname}" if dir != "" else f"{prefix}_{fname}"
        return out_path